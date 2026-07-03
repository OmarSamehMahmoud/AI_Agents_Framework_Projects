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

    mkdir CaracalX_Researcher_ChatBot
    cd CaracalX_Researcher_ChatBot
    git init
    git remote add -f origin https://github.com/OmarSamehMahmoud/AI_Agents_Framework_Projects.git
    git config core.sparseCheckout true
    echo "CaracalX_Researcher_ChatBot/" >> .git/info/sparse-checkout
    git pull origin master
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

---

## 🛠️ Native Tools (4 Tools)

CaracalX includes **4 built-in tools** for core functionality:

* **calculator** — Evaluate mathematical expressions safely using a restricted set of math functions.
* **get_stock_price** — Fetch real-time stock market data and ticker information via AlphaVantage API.
* **web_search** — Perform deep web research using Tavily API and return synthesized results from top sources.
* **rag_tool** — Retrieve relevant information from uploaded PDF documents using thread-specific FAISS vector store.

---

## 🔌 MCP Tools (20 Tools)

CaracalX integrates **20 specialized tools** via the Model Context Protocol (MCP), organized into two dedicated servers:

### 📊 Expense Tracker Server
* **add_expense** — Add a new expense with amount, category, and description.
* **list_expenses** — Return all recorded expenses.
* **total_expenses** — Calculate the total sum of all expenses.
* **category_summary** — Show spending breakdown by category.
* **delete_expense** — Delete an expense by its index.

### 🧮 Advanced Math Server (Powered by SymPy)
**Algebra & Calculus:**
* **evaluate** — Evaluate a mathematical expression (e.g., `2 + 3*4`).
* **simplify_expression** — Simplify algebraic expressions.
* **factor_expression** — Factor algebraic expressions (e.g., `x² - 9`).
* **expand_expression** — Expand algebraic expressions (e.g., `(x+2)*(x+3)`).
* **solve_equation** — Solve equations for a given variable.
* **derivative** — Compute the derivative of an expression.
* **integral** — Compute the indefinite integral of an expression.

**Linear Algebra:**
* **matrix_multiply** — Multiply two matrices.
* **matrix_determinant** — Calculate the determinant of a matrix.

**Statistics:**
* **mean** — Calculate the arithmetic mean.
* **median** — Calculate the median.
* **standard_deviation** — Calculate the standard deviation.

**Number Theory:**
* **prime_factors** — Return the prime factorization of a number.
* **gcd** — Calculate the greatest common divisor.
* **lcm** — Calculate the least common multiple.

**Total: 24 Tools (4 Native + 20 MCP)**
---

### 6. ScreenShots
<img width="954" height="440" alt="image" src="https://github.com/user-attachments/assets/dc859e24-4b70-44ae-9166-85916d874b56" />
<img width="950" height="439" alt="image" src="https://github.com/user-attachments/assets/3bd28f72-39df-4897-9239-10c8a4434097" />
<img width="950" height="431" alt="image" src="https://github.com/user-attachments/assets/b0a6f133-7495-442e-8e6f-4a63ddec1138" />
<img width="950" height="423" alt="image" src="https://github.com/user-attachments/assets/8ef88ec2-e367-485c-ba46-f6fd8b3c49cf" />
<img width="948" height="434" alt="image" src="https://github.com/user-attachments/assets/bef68249-8254-4e0a-8e86-621cd24d24f6" />
<img width="950" height="437" alt="image" src="https://github.com/user-attachments/assets/1dada73f-550e-4b65-bdb9-2d3cab5cf393" />

---

## 📝 License

This project is proprietary and confidential.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---
