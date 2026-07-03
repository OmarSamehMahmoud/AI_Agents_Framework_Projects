"""
Run this ONCE to download Qwen 3.5 0.8B into ./qwen35_model/
  python download_model.py
or with your token:
  HF_TOKEN=hf_xxx python download_model.py
"""

import os
import sys
from pathlib import Path

MODEL_ID   = "Qwen/Qwen3.5-0.8B"
SAVE_DIR   = Path(__file__).parent / "qwen35_model"
HF_TOKEN   = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""

def main():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("❌  Missing packages.  Run:  pip install -r requirements.txt")
        sys.exit(1)

    if SAVE_DIR.exists() and any(SAVE_DIR.iterdir()):
        print(f"✅  Model already downloaded at {SAVE_DIR}")
        print("    Delete the folder and re-run if you want a fresh download.")
        return

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    token_arg = {"token": HF_TOKEN} if HF_TOKEN else {}
    if not HF_TOKEN:
        print("⚠️   HF_TOKEN not set — this will only work if the model is public.")
        print("    Set it with:  export HF_TOKEN=hf_your_token_here\n")

    print(f"⬇️   Downloading {MODEL_ID}  →  {SAVE_DIR}")
    print("    This should be relatively quick...\n")

    # ── Tokenizer ──────────────────────────────────────────────
    print("  [1/2] Downloading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **token_arg)
    tokenizer.save_pretrained(SAVE_DIR)
    print("  ✓ Tokenizer saved\n")

    # ── Model weights ──────────────────────────────────────────
    print("  [2/2] Downloading model weights…")
    print("        (Qwen 3.5 0.8B is ~1.6 GB in bfloat16 — should be quick ☕)\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,   # half-precision → cuts RAM roughly in half
        device_map="auto",             # GPU if available, else CPU
        **token_arg,
    )
    model.save_pretrained(SAVE_DIR)
    print(f"\n✅  Model saved to  {SAVE_DIR}")
    print("    You can now run:  python app.py")


if __name__ == "__main__":
    main()