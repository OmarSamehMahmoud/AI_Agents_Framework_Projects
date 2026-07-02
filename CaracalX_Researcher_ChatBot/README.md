# 🐆 CaracalX Researcher

**Uncover. Analyze. Master.**

CaracalX Researcher is an advanced, locally-hosted AI research assistant featuring a premium dark glassmorphism UI. Powered by a custom LangGraph agent, it performs deep web research, analyzes uploaded PDFs via a local RAG pipeline, executes complex multi-step tool chains, and maintains persistent memory across sessions.

---

## 🚀 Key Features

### 🧠 Dual-Memory Architecture (LangGraph)
* **Short-Term (Episodic) Memory:** Managed via a hybrid trim-and-summarization engine. It keeps the most recent interactions intact while automatically compressing older conversation history into concise summaries to prevent context overflow and maintain high reasoning accuracy.
* **Long-Term Memory (LTM):** A dedicated PostgreSQL-backed memory core that continuously extracts and stores atomic facts about the user (preferences, identity, ongoing projects). The agent recalls these details in future sessions to provide highly personalized interactions.

### 📄 Local RAG Pipeline
* Securely ingests and indexes uploaded PDF documents using **FAISS** and **HuggingFace Embeddings** (`BAAI/bge-small-en-v1.5`).
* Features a thread-specific retriever that ensures document context is strictly isolated per chat session.
* Includes robust error handling for scanned/image-only PDFs, prompting the user to run OCR if necessary.

### 🔌 Tool Orchestration & MCP (Model Context Protocol)
* Native support for the **Model Context Protocol (MCP)**, allowing seamless integration of external, asynchronous tool servers (e.g., Math, Expense tracking) via `stdio` transport.
* Dynamically binds and converts complex LangChain/MCP tool schemas into Ollama-compatible JSON formats on the fly.
* Custom-built tool execution loop with strict state sanitization to prevent graph crashes and malformed tool histories.

### 🌐 Real-time Web & Market Data
* **Deep Web Research:** Powered by **Tavily** for fast, accurate, and up-to-date web scraping and search synthesis.
* **Financial Tracking:** Integrated with **AlphaVantage** to fetch real-time stock market data and ticker information.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Custom styled with a premium dark glassmorphism theme, asynchronous streaming simulation).
* **Agent Framework:** LangGraph & LangChain (Custom StateGraph with conditional tool routing and checkpointing).
* **LLM Inference:** Ollama (Model: `gpt-oss:20b-cloud`).
  * *Note: Uses a custom `BaseChatModel` wrapper to interact with Ollama via direct HTTP `requests` to the `/api/chat` endpoint, bypassing heavy native SDKs and ensuring zero timeout restrictions.*
* **Databases:**
  * **PostgreSQL:** Long-term Memory Store (via `langgraph.store.postgres`).
  * **SQLite:** Short-term Session State & Checkpointing (via `langgraph.checkpoint.sqlite`).
  * **FAISS:** High-performance Vector Database for Local RAG.
* **Integrations:** Model Context Protocol (MCP), Tavily API, AlphaVantage API.

---

## ⚙️ Local Setup Instructions

### Prerequisites
* **Python 3.10+**
* **Ollama** installed and running locally.
* **PostgreSQL** database running (default URI configured for `localhost:5432`).
* API Keys for **Tavily** and **AlphaVantage**.

### 1. Clone and Install Dependencies

    git clone <your-repo-url>
    cd CaracalX_Researcher
    pip install -r requirements.txt

### 2. Configure Ollama
Ensure the Ollama service is running in the background, then pull the required model:

    ollama pull gpt-oss:20b-cloud

### 3. Environment Variables
Create a `.env` file in the root directory and add your API keys:

    TAVILY_API_KEY=your_tavily_api_key
    ALPHAVANTAGE_API_KEY=your_alphavantage_api_key

### 4. Database Setup
Ensure your PostgreSQL server is running. The application will automatically attempt to create the required `store` and `store_migrations` tables on startup.
*(If you need to configure a different database URI, update the `DB_URI` variable in `backend.py`)*.

### 5. Run the Application
Launch the Streamlit interface:

    python -m streamlit run app.py

Navigate to `http://localhost:8501` in your browser to start hunting.

### 6. ScreenShots
<img width="947" height="433" alt="image" src="https://github.com/user-attachments/assets/05cb2459-985b-49db-bcd1-1cc7d82097fb" />
<img width="949" height="409" alt="image" src="https://github.com/user-attachments/assets/6d66d752-6331-4081-b118-49fcfdc0bcb9" />
<img width="941" height="421" alt="image" src="https://github.com/user-attachments/assets/a6351f50-2191-4f91-b9bd-2798bd02faa9" />
<img width="944" height="413" alt="image" src="https://github.com/user-attachments/assets/68553707-0bab-4ddd-a963-5711957bd066" />
<img width="953" height="410" alt="image" src="https://github.com/user-attachments/assets/437aafe2-9a73-4344-a11e-5d34040038be" />
<img width="949" height="404" alt="image" src="https://github.com/user-attachments/assets/2bfb1d3b-f745-4f96-9a5b-06fc92f8f1dd" />



---

## 📝 License

This project is proprietary and confidential.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---
