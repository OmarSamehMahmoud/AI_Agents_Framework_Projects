# SQL Agent — Qwen3.5:0.8b (Local)

Natural language → SQL, powered by a **locally downloaded** Qwen3.5:0.8b model.
No API calls after setup. Everything runs on your machine.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Qwen3.5-0.8B into ./qwen_model/  (one-time, ~1.6 GB)
export HF_TOKEN=hf_your_token_here
python download_model.py

# 3. Start the server
python app.py
```

Open http://localhost:5000

## Project layout

```
sql_agent/
├── app.py               ← Flask backend  (loads model locally)
├── download_model.py    ← One-time model downloader
├── templates/
│   └── index.html       ← Chat UI (pure HTML/CSS/JS)
├── requirements.txt
├── company.db           ← Auto-created on first run
└── qwen_model/          ← Created by download_model.py (~1.6 GB)
    ├── config.json
    ├── tokenizer.json
    └── model-*.safetensors
```

## Hardware requirements

| Setup | Minimum RAM | Notes |
|-------|-------------|-------|
| GPU (CUDA) | 4 GB VRAM | fp16; very fast inference |
| CPU only | 8 GB RAM | Moderate speed (~2–10 s per query) |
| Apple M-chip | 8 GB unified | MPS backend; good speed |

**Note:** Qwen3.5-0.8B is a lightweight model (~1.6 GB), so it runs smoothly on most modern machines without requiring high-end hardware.

## How to get a free HF token

1. Sign up at https://huggingface.co
2. Settings → Access Tokens → New token (Read)
3. Visit https://huggingface.co/Qwen/Qwen3.5-0.8B → Request access
4. `export HF_TOKEN=hf_...`

## Example questions

- Show all employees
- Who has the highest salary?
- List all employees in the IT department
- What is the average salary by department?
- Which projects are in progress?
- Show employees assigned to the Mobile App Rewrite project

## ScreenShots

<img width="956" height="440" alt="image" src="https://github.com/user-attachments/assets/33177864-d8b2-442b-a77e-84bb07b88b33" />
<img width="955" height="439" alt="image" src="https://github.com/user-attachments/assets/73d4d161-4764-4576-83fd-9620a1fae19e" />
