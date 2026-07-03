# SQL Agent — Gemma 4 E4B (Local)

Natural language → SQL, powered by a **locally downloaded** Gemma 4 model.
No API calls after setup. Everything runs on your machine.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Gemma 4 into ./gemma4_model/  (one-time, ~9 GB)
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
└── gemma4_model/        ← Created by download_model.py (~9 GB)
    ├── config.json
    ├── tokenizer.json
    └── model-*.safetensors
```

## Hardware requirements

| Setup        | Minimum RAM | Notes                              |
|--------------|-------------|-------------------------------------|
| GPU (CUDA)   | 10 GB VRAM  | bfloat16; fast inference            |
| CPU only     | 20 GB RAM   | Slow (~30–120 s per query)          |
| Apple M-chip | 16 GB RAM   | MPS backend; decent speed           |

## How to get a free HF token

1. Sign up at https://huggingface.co
2. Settings → Access Tokens → New token (Read)
3. Visit https://huggingface.co/google/gemma-4-e4b-it → Request access
4. `export HF_TOKEN=hf_...`

## Example questions

- Show all employees
- Who has the highest salary?
- List all employees in the IT department
- What is the average salary by department?
- Which projects are in progress?
- Show employees assigned to the Mobile App Rewrite project
