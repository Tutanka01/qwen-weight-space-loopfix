"""Load the LM Studio Qwen3.5-0.8B GGUF (Q8_0) into a transformers Qwen3_5ForCausalLM.

transformers cannot load `qwen35` GGUF files itself, so we dequantize every tensor and
undo the transforms llama.cpp's converter applies:
  * RMSNorm weights are stored as (1 + w)        -> subtract 1 (except the gated DeltaNet norm)
  * A_log is stored as ssm_a = -exp(A_log)       -> A_log = log(-ssm_a)
  * conv1d weight is stored squeezed [C, K]      -> [C, 1, K]
"""

from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

from loopfix.paths import BASE_GGUF as DEFAULT_GGUF


def field(reader: GGUFReader, key: str):
    return reader.fields[key].contents()


def read_tensors(reader: GGUFReader) -> dict[str, np.ndarray]:
    out = {}
    for t in reader.tensors:
        arr = dequantize(t.data, t.tensor_type).astype(np.float32)
        out[t.name] = arr.reshape([int(x) for x in reversed(t.shape)])
    return out


def hf_state_dict(gg: dict[str, np.ndarray], n_layers: int) -> dict[str, torch.Tensor]:
    sd = {}
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    sd["model.embed_tokens.weight"] = t(gg["token_embd.weight"])
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]  # tied embeddings (no output.weight in GGUF)
    sd["model.norm.weight"] = t(gg["output_norm.weight"] - 1)
    for i in range(n_layers):
        b, p = f"blk.{i}.", f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = t(gg[b + "attn_norm.weight"] - 1)
        sd[p + "post_attention_layernorm.weight"] = t(gg[b + "post_attention_norm.weight"] - 1)
        for g, h in [("ffn_gate", "gate_proj"), ("ffn_up", "up_proj"), ("ffn_down", "down_proj")]:
            sd[p + f"mlp.{h}.weight"] = t(gg[b + g + ".weight"])
        if b + "attn_q.weight" in gg:  # full (gated) attention layer
            a = p + "self_attn."
            sd[a + "q_proj.weight"] = t(gg[b + "attn_q.weight"])
            sd[a + "k_proj.weight"] = t(gg[b + "attn_k.weight"])
            sd[a + "v_proj.weight"] = t(gg[b + "attn_v.weight"])
            sd[a + "o_proj.weight"] = t(gg[b + "attn_output.weight"])
            sd[a + "q_norm.weight"] = t(gg[b + "attn_q_norm.weight"] - 1)
            sd[a + "k_norm.weight"] = t(gg[b + "attn_k_norm.weight"] - 1)
        else:  # gated DeltaNet layer
            a = p + "linear_attn."
            sd[a + "in_proj_qkv.weight"] = t(gg[b + "attn_qkv.weight"])
            sd[a + "in_proj_z.weight"] = t(gg[b + "attn_gate.weight"])
            sd[a + "in_proj_b.weight"] = t(gg[b + "ssm_beta.weight"])
            sd[a + "in_proj_a.weight"] = t(gg[b + "ssm_alpha.weight"])
            sd[a + "conv1d.weight"] = t(gg[b + "ssm_conv1d.weight"][:, None, :])
            sd[a + "dt_bias"] = t(gg[b + "ssm_dt.bias"])
            sd[a + "A_log"] = t(np.log(-gg[b + "ssm_a"]))
            sd[a + "norm.weight"] = t(gg[b + "ssm_norm.weight"])
            sd[a + "out_proj.weight"] = t(gg[b + "ssm_out.weight"])
    return sd


def build_config(reader: GGUFReader):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    f = lambda k: field(reader, "qwen35." + k)
    n_layers = int(f("block_count"))
    interval = int(f("full_attention_interval"))
    head_dim = int(f("attention.key_length"))
    n_v_heads = int(f("ssm.time_step_rank"))
    return Qwen3_5TextConfig(
        vocab_size=len(field(reader, "tokenizer.ggml.tokens")),
        hidden_size=int(f("embedding_length")),
        intermediate_size=int(f("feed_forward_length")),
        num_hidden_layers=n_layers,
        num_attention_heads=int(f("attention.head_count")),
        num_key_value_heads=int(f("attention.head_count_kv")),
        head_dim=head_dim,
        rms_norm_eps=float(f("attention.layer_norm_rms_epsilon")),
        linear_conv_kernel_dim=int(f("ssm.conv_kernel")),
        linear_key_head_dim=int(f("ssm.state_size")),
        linear_value_head_dim=int(f("ssm.inner_size")) // n_v_heads,
        linear_num_key_heads=int(f("ssm.group_count")),
        linear_num_value_heads=n_v_heads,
        layer_types=["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_layers)],
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(f("rope.freq_base")),
            "partial_rotary_factor": int(f("rope.dimension_count")) / head_dim,
            "mrope_section": [int(x) for x in f("rope.dimension_sections")][:3],
            "mrope_interleaved": True,
        },
        eos_token_id=int(field(reader, "tokenizer.ggml.eos_token_id")),
        pad_token_id=int(field(reader, "tokenizer.ggml.padding_token_id")) if "tokenizer.ggml.padding_token_id" in reader.fields else None,
    )


def build_tokenizer(reader: GGUFReader):
    """Byte-level BPE tokenizer rebuilt from the GGUF vocab/merges (Qwen2-style pre-tokenizer)."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokens = field(reader, "tokenizer.ggml.tokens")
    merges = [tuple(m.split(" ", 1)) for m in field(reader, "tokenizer.ggml.merges")]
    types = field(reader, "tokenizer.ggml.token_type")
    vocab = {tok: i for i, tok in enumerate(tokens)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=False))
    tk.normalizer = normalizers.NFC()
    pat = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pat), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    # control / user-defined tokens (types 3 and 4) must never be split
    from tokenizers import AddedToken
    special = [AddedToken(tokens[i], special=True, normalized=False) for i, ty in enumerate(types) if ty in (3, 4)]
    tk.add_special_tokens(special)
    eos = tokens[int(field(reader, "tokenizer.ggml.eos_token_id"))]
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token=eos,
        pad_token=tokens[int(field(reader, "tokenizer.ggml.padding_token_id"))] if "tokenizer.ggml.padding_token_id" in reader.fields else eos,
        chat_template=field(reader, "tokenizer.chat_template"),
    )


def load(gguf_path: Path | str = DEFAULT_GGUF, dtype=torch.float32, device: str = "cpu"):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    reader = GGUFReader(str(gguf_path))
    config = build_config(reader)
    sd = hf_state_dict(read_tensors(reader), config.num_hidden_layers)
    model = Qwen3_5ForCausalLM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"missing={missing[:10]} unexpected={unexpected[:10]}")
    model.tie_weights()
    model = model.to(dtype=dtype, device=device).eval()
    return model, build_tokenizer(reader)
