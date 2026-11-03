"""Write an edited copy of the GGUF: only the ffn_down tensors of edited layers are re-quantized
(Q8_0) and overwritten in place; every other byte of the file is identical to the original.

Edit spec (JSON): {"mode": "zero" | "proj", "layers": {"19": [neuron, ...], ...}}
  zero : W_down[:, j] = 0
  proj : W_down[:, j] -= (d_l . W_down[:, j]) d_l   with d_l from results/acts/dirs.npz
Usage: uv run write_gguf.py edit.json models/out.gguf
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize, quantize

from qwen_gguf import DEFAULT_GGUF


def edited_tensor(t, mode, neurons, d):
    shape = [int(x) for x in reversed(t.shape)]  # [hidden=1024, inter=3584]
    W = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
    W0 = W.copy()
    idx = np.array(neurons)
    if mode == "zero":
        W[:, idx] = 0
    else:
        cols = W[:, idx]
        W[:, idx] = cols - np.outer(d, d @ cols)
    q = quantize(W, t.tensor_type)
    assert q.nbytes == t.n_bytes, (q.nbytes, t.n_bytes)
    # sanity: Q8_0 blocks hold 32 consecutive neurons sharing one scale. Columns outside the edited
    # blocks must be bit-identical; neighbours inside an edited block may move by < 1 quantization step.
    back = dequantize(q, t.tensor_type).reshape(shape)
    touched = np.zeros(shape[1], bool)
    for j in idx:
        touched[(j // 32) * 32:(j // 32 + 1) * 32] = True
    touched_other = touched.copy()
    touched_other[idx] = False
    assert np.array_equal(back[:, ~touched], W0[:, ~touched])
    if touched_other.any():
        step = np.abs(W0).max() / 127
        assert np.abs(back[:, touched_other] - W0[:, touched_other]).max() <= step
    return q


def main(spec_path, out_path):
    spec = json.loads(Path(spec_path).read_text())
    dirs = np.load("results/acts/dirs.npz")
    out = Path(out_path)
    shutil.copyfile(DEFAULT_GGUF, out)
    reader = GGUFReader(str(out))
    tensors = {t.name: t for t in reader.tensors}
    with out.open("r+b") as f:
        for l, neurons in spec["layers"].items():
            t = tensors[f"blk.{l}.ffn_down.weight"]
            q = edited_tensor(t, spec["mode"], neurons, dirs[str(l)].astype(np.float32))
            f.seek(t.data_offset)
            f.write(q.tobytes())
            print(f"layer {l}: {len(neurons)} neurons edited ({spec['mode']})")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
