"""All repository paths in one place, so scripts work from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EDITS = ROOT / "edits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
BASELINE = RESULTS / "baseline"  # LM Studio traces of the original model (all 60 prompts)
HELDOUT = RESULTS / "heldout"    # final evaluation on the 30 test prompts
SEARCH = RESULTS / "search"      # proxy scores of candidate edits
ACTS = RESULTS / "acts"          # activations / directions / neuron statistics

PROMPTS = DATA / "prompts_fr.jsonl"
PROMPTS_TEST = DATA / "prompts_fr_test.jsonl"
BASE_GGUF = MODELS / "Qwen3.5-0.8B-Q8_0.gguf"
TRACES = [BASELINE / "default.jsonl", BASELINE / "qwen_reco.jsonl"]  # used by the analysis scripts
