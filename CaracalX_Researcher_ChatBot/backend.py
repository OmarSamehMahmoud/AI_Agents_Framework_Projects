# ==============================================================================
# LIBRARIES AND DEPENDENCIES IMPORTS
# ==============================================================================
import uuid
import psycopg
from psycopg_pool import ConnectionPool
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
import os
import math
from langchain_core.runnables import RunnableConfig
import sqlite3
import asyncio
import threading
from typing import TypedDict, Annotated, Optional, Any, Dict, List
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import requests
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys
import json

# ==============================================================================
# CONCURRENCY AND MCP INTERFACE
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
math_mcp_path = os.path.join(BASE_DIR, "mcp_servers", "math_mcp", "main.py")
expense_mcp_path = os.path.join(BASE_DIR, "mcp_servers", "expense_mcp", "main.py")

client = MultiServerMCPClient(
    {
        'arith': {  
            'transport': 'stdio',
            'command': sys.executable, 
            'args': ["-u", math_mcp_path] 
        },
        'expense': { 
            'transport': 'stdio',
            'command': sys.executable,
            'args': ["-u", expense_mcp_path]
        }
    }
)

mcp_tools = []
_mcp_loop = asyncio.new_event_loop()

def _run_mcp_loop():
    asyncio.set_event_loop(_mcp_loop)
    _mcp_loop.run_forever()

_mcp_thread = threading.Thread(target=_run_mcp_loop, daemon=True)
_mcp_thread.start()

try:
    print("🔌 Initializing MCP servers...")
    future = asyncio.run_coroutine_threadsafe(client.get_tools(), _mcp_loop)
    _mcp_tools_async = future.result()
    
    for t in _mcp_tools_async:
        def create_sync_func(async_tool):
            def sync_func(*args, **kwargs):
                tool_input = kwargs if kwargs else (args[0] if args else {})
                fut = asyncio.run_coroutine_threadsafe(async_tool.ainvoke(tool_input), _mcp_loop)
                result = fut.result()
                if not result:
                    return "Operation completed successfully, but no data was returned."
                return str(result)
            return sync_func

        mcp_tools.append(
            StructuredTool(
                name=t.name,
                description=t.description,
                args_schema=t.args_schema,
                func=create_sync_func(t)
            )
        )
    print(f"✅ Successfully loaded {len(mcp_tools)} MCP tools.")
except Exception as e:
    print(f"⚠️ Warning: Failed to load MCP tools. Reason: {e}")
    mcp_tools = []

# ==============================================================================
# LOCAL RAG MANAGEMENT STORES
# ==============================================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_docs = [
            doc for doc in docs
            if isinstance(doc.page_content, str) and doc.page_content.strip()
        ]

        if not text_docs:
            raise ValueError(
                "No readable text was found in this PDF. It may be scanned, "
                "image-only, blank, or protected. Run OCR on the PDF and upload it again."
            )
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(text_docs)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        if not chunks:
            raise ValueError("The PDF text could not be split into indexable chunks.")
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        
        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try: os.remove(temp_path)
        except OSError: pass

# ==============================================================================
# LONG-TERM MEMORY (POSTGRES STORE INITIALIZATION)
# ==============================================================================

DB_URI = "postgresql://postgres:Omar1996@localhost:5432/postgres?sslmode=disable"
pool = ConnectionPool(conninfo=DB_URI)
store = PostgresStore(pool)

print("🗄️ Initializing Postgres Long-Term Memory store...")
try:
    store.setup()
    print("✅ Postgres store initialized successfully!")
except Exception as e:
    print(f"⚠️ LangGraph Postgres setup hit an issue: {e}")
    print("   Fixing manually...")
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    prefix text NOT NULL,
                    key text NOT NULL,
                    value jsonb NOT NULL,
                    created_at timestamptz DEFAULT now(),
                    updated_at timestamptz DEFAULT now(),
                    PRIMARY KEY (prefix, key)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS store_migrations (
                    v integer PRIMARY KEY
                )
            """)
            
            cur.execute("SELECT COALESCE(MAX(v), -1) FROM store_migrations")
            current_version = cur.fetchone()[0]
            
            for v, sql in enumerate(store.MIGRATIONS[current_version + 1:], start=current_version + 1):
                safe_sql = sql.replace("CONCURRENTLY ", "")
                try:
                    cur.execute(safe_sql)
                    cur.execute("INSERT INTO store_migrations (v) VALUES (%s) ON CONFLICT DO NOTHING", (v,))
                except Exception as mig_err:
                    print(f"   Warning on migration {v}: {mig_err}")
                    
    print("✅ Postgres store initialized successfully!")

# ==============================================================================
# NATIVE TOOLS & PYDANTIC SCHEMAS
# ==============================================================================

class SearchInput(BaseModel):
    query: str = Field(description="A very short, simple search query. Maximum 3 to 5 words. Do not use punctuation.")

class StockInput(BaseModel):
    symbol: str = Field(description="The exact stock ticker symbol. Example: AAPL")

class CalcInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate.")

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this memory is NEW and should be stored. False if duplicate/already known.")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")

# ==============================================================================
# MODEL INITIALIZATION (OLLAMA CUSTOM CHAT)
# ==============================================================================

print("🧠 Connecting to local Ollama instance...")

class OllamaCustomChat(BaseChatModel):
    """Custom LangChain Chat Model wrapper for Ollama using direct HTTP requests."""
    model_name: str
    base_url: str = "http://localhost:11434"
    
    @property
    def _llm_type(self) -> str:
        return "ollama-custom"

    def bind_tools(self, tools: list, **kwargs):
        """
        Bind tools to the model. LangChain's BaseChatModel raises NotImplementedError 
        for this by default, so we must implement it using the standard Runnable .bind() method.
        """
        return self.bind(tools=tools, **kwargs)

    def _convert_messages_to_ollama(self, messages: list[BaseMessage]) -> list[dict]:
        ollama_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                ollama_msgs.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                ollama_msgs.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    msg_dict["tool_calls"] = []
                    for tc in msg.tool_calls:
                        # CRITICAL FIX 1: Ollama expects 'arguments' to be a dictionary, NOT a JSON string!
                        args = tc.get("args", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                                
                        # CRITICAL FIX 2: Ollama's request schema does NOT allow an 'id' field here.
                        msg_dict["tool_calls"].append({
                            "function": {
                                "name": tc["name"],
                                "arguments": args
                            }
                        })
                ollama_msgs.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                tool_msg = {
                    "role": "tool", 
                    "content": str(msg.content) if msg.content is not None else "",
                    "name": getattr(msg, "name", "") or ""
                }
                # Ollama supports tool_call_id in the tool response to map it back
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    tool_msg["tool_call_id"] = msg.tool_call_id
                ollama_msgs.append(tool_msg)
        return ollama_msgs
        
    def _convert_tools_to_ollama(self, tools: list) -> list[dict]:
        ollama_tools = []
        for t in tools:
            # Safely get the tool name and description
            tool_name = getattr(t, "name", "unknown_tool")
            tool_desc = getattr(t, "description", "")
            
            # Initialize an empty schema
            schema = {"type": "object", "properties": {}, "required": []}
            
            # Try to extract a proper schema only if the tool is a standard LangChain tool
            try:
                if hasattr(t, "args_schema") and t.args_schema:
                    if hasattr(t.args_schema, "model_json_schema"):
                        schema = t.args_schema.model_json_schema()
                    elif hasattr(t.args_schema, "schema"):
                        schema = t.args_schema.schema()
                    else:
                        # Fallback for unknown schema types
                        schema = {"type": "object", "properties": {}, "required": []}
                elif hasattr(t, "get_input_schema"):
                    inp_schema = t.get_input_schema()
                    if hasattr(inp_schema, "model_json_schema"):
                        schema = inp_schema.model_json_schema()
                    elif hasattr(inp_schema, "schema"):
                        schema = inp_schema.schema()
                    else:
                        schema = {"type": "object", "properties": {}, "required": []}
                # If none of the above, we keep the empty schema
            except Exception as schema_err:
                print(f"[Ollama Tool Conversion] Warning: Could not extract schema for tool '{tool_name}': {schema_err}")
                schema = {"type": "object", "properties": {}, "required": []}
            
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": schema
                }
            })
        return ollama_tools

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        ollama_msgs = self._convert_messages_to_ollama(messages)
        payload = {
            "model": self.model_name,
            "messages": ollama_msgs,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1)
            }
        }
        
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = self._convert_tools_to_ollama(tools)
            
        print(f"\n[DEBUG OLLAMA] Sending payload to /api/chat")
        
        response = requests.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        res_data = response.json()
        
        print(f"[DEBUG OLLAMA] Received response: {json.dumps(res_data, indent=2)}")
        
        message_data = res_data.get("message", {})
        content = message_data.get("content", "")
        tool_calls_data = message_data.get("tool_calls", [])
        
        tool_calls = []
        if tool_calls_data:
            for i, tc in enumerate(tool_calls_data):
                func = tc.get("function", {})
                name = func.get("name", "unknown_tool")
                args_raw = func.get("arguments", "{}")
                
                # CRITICAL FIX: Handle both string and dict arguments
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(args_raw, dict):
                    args = args_raw
                else:
                    args = {}
                    
                tool_calls.append({
                    "name": name,
                    "args": args,
                    "id": tc.get("id", f"call_{i}_{name}")
                })
                
        ai_msg = AIMessage(content=content, tool_calls=tool_calls if tool_calls else [])
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        ollama_msgs = self._convert_messages_to_ollama(messages)
        payload = {
            "model": self.model_name,
            "messages": ollama_msgs,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.1)
            }
        }
        
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = self._convert_tools_to_ollama(tools)
            
        print(f"\n[DEBUG OLLAMA STREAM] Sending payload to /api/chat")
            
        yielded_something = False
        
        try:
            with requests.post(f"{self.base_url}/api/chat", json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        res_data = json.loads(line)
                        # DEBUG: Uncomment the next line to see every raw chunk from Ollama
                        # print(f"[DEBUG OLLAMA STREAM] Raw chunk: {res_data}")
                        
                        message_data = res_data.get("message", {})
                        content = message_data.get("content", "")
                        is_done = res_data.get("done", False)
                        
                        if content:
                            yield ChatGenerationChunk(message=AIMessageChunk(content=content))
                            yielded_something = True
                            
                        if is_done:
                            tool_calls_data = message_data.get("tool_calls", [])
                            if tool_calls_data:
                                tool_calls = []
                                for i, tc in enumerate(tool_calls_data):
                                    func = tc.get("function", {})
                                    name = func.get("name", "unknown_tool")
                                    args_raw = func.get("arguments", "{}")
                                    
                                    # CRITICAL FIX: Handle both string and dict arguments
                                    if isinstance(args_raw, str):
                                        try:
                                            args = json.loads(args_raw)
                                        except json.JSONDecodeError:
                                            args = {}
                                    elif isinstance(args_raw, dict):
                                        args = args_raw
                                    else:
                                        args = {}
                                        
                                    tool_calls.append({
                                        "name": name,
                                        "args": args,
                                        "id": tc.get("id", f"call_{i}_{name}")
                                    })
                                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=tool_calls))
                                yielded_something = True
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[Ollama Stream Error] {e}")
            if not yielded_something:
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"[Stream Error: {e}]"))
                yielded_something = True

        if not yielded_something:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            
# Create model instances using your Ollama model
model = OllamaCustomChat(model_name="gpt-oss:20b-cloud")
memory_llm = OllamaCustomChat(model_name="gpt-oss:20b-cloud")

# Try to set up structured output (may not work with all models)
try:
    memory_extractor = memory_llm.with_structured_output(MemoryDecision)
    print("✅ Structured output enabled for memory extraction")
except Exception as e:
    print(f"⚠️ Structured output not supported, using JSON fallback: {e}")
    memory_extractor = memory_llm 

@tool(args_schema=CalcInput)
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    allowed_names = {
        "sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round,
        "ceil": math.ceil, "floor": math.floor, "sin": math.sin,
        "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool(args_schema=StockInput)
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock price. Input must be a stock ticker symbol (e.g., AAPL, TSLA, GOOG)."""
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&interval=5min&apikey=JYMA75DC485CCPUF'
    r = requests.get(url)
    return r.json()

@tool(args_schema=SearchInput)
def web_search(query: str) -> str:
    """Search the web for current information and return the top search results."""
    try:
        tavily = TavilyClient()
        response = tavily.search(query=query, max_results=5, search_depth="basic")
        output = []
        for result in response.get("results", []):
            output.append(f"Title: {result.get('title')}\nContent: {result.get('content')}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"

# 1. Define a strict Pydantic schema for the tool
class RagToolInput(BaseModel):
    query: str = Field(description="The search query to find relevant information in the uploaded PDF.")
    thread_id: str = Field(description="The exact chat thread ID provided in the system prompt.")

# 2. Apply the schema to the tool
@tool(args_schema=RagToolInput)
def rag_tool(query: str, thread_id: str) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread. Use this whenever the user asks about an attached document or PDF."""
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}
    
    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

# ==============================================================================
# STATE PERSISTENCE (SHORT-TERM EPISODIC MEMORY)
# ==============================================================================

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS thread_metadata (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL
)
""")
conn.commit()

cheakpoint = SqliteSaver(conn=conn)

class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==============================================================================
# TOOLS BINDING
# ==============================================================================

tools = [web_search, get_stock_price, calculator, rag_tool] + mcp_tools
llm_with_tools = model.bind_tools(tools)

# ==============================================================================
# CONTEXT OVERFLOW MANAGEMENT — HYBRID TRIM + SUMMARIZATION
# ==============================================================================

_THREAD_SUMMARIES: Dict[str, str] = {}
RECENT_MESSAGES_TO_KEEP = 6
SUMMARIZATION_TRIGGER = 14
HARD_TRIM_LIMIT = 20

def _classify_message_role(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage): return "User"
    if isinstance(msg, AIMessage): return "Assistant"
    if isinstance(msg, ToolMessage): return f"Tool({msg.name})"
    return "System"

def _extract_text_content(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        parts = [block.get("text", "").strip() for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return " ".join(parts)
    return str(content).strip()

def _build_summary_prompt(messages_to_summarize: List[BaseMessage], existing_summary: Optional[str]) -> str:
    lines = []
    for msg in messages_to_summarize:
        role = _classify_message_role(msg)
        text = _extract_text_content(msg)
        if text: lines.append(f"{role}: {text[:400]}")
    transcript = "\n".join(lines)

    if existing_summary:
        return (
            "You are a concise summarizer. Below is a prior summary of an ongoing conversation, "
            "followed by additional messages that occurred after it. "
            "Produce a single updated summary in under 150 words. "
            "Preserve key facts, decisions, numbers, and named entities. "
            "Do not add opinions or padding.\n\n"
            f"Prior summary:\n{existing_summary}\n\n"
            f"New messages:\n{transcript}\n\n"
            "Updated summary:"
        )
    else:
        return (
            "You are a concise summarizer. Below is a portion of a conversation. "
            "Produce a summary in under 150 words. "
            "Preserve key facts, decisions, numbers, and named entities. "
            "Do not add opinions or padding.\n\n"
            f"Conversation:\n{transcript}\n\n"
            "Summary:"
        )

def summarize_messages(messages_to_summarize: List[BaseMessage], thread_id: str) -> str:
    existing_summary = _THREAD_SUMMARIES.get(thread_id)
    prompt = _build_summary_prompt(messages_to_summarize, existing_summary)
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        new_summary = response.content.strip()
        _THREAD_SUMMARIES[thread_id] = new_summary
        return new_summary
    except Exception as e:
        print(f"[ContextManager] Summarization failed for thread {thread_id}: {e}")
        return existing_summary or ""

def manage_context(messages: List[BaseMessage], thread_id: str) -> List[BaseMessage]:
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system_msgs) > SUMMARIZATION_TRIGGER:
        older_slice = non_system_msgs[:-RECENT_MESSAGES_TO_KEEP]
        recent_slice = non_system_msgs[-RECENT_MESSAGES_TO_KEEP:]
        summarize_messages(older_slice, thread_id)
        non_system_msgs = recent_slice

    if len(non_system_msgs) > HARD_TRIM_LIMIT:
        excess = len(non_system_msgs) - HARD_TRIM_LIMIT
        non_system_msgs = non_system_msgs[excess:]

    summary = _THREAD_SUMMARIES.get(thread_id)
    summary_injection = []
    if summary:
        summary_injection = [
            SystemMessage(content=(
                "The following is a running summary of the conversation history that "
                "occurred before this session window. Use it as context:\n\n"
                f"{summary}"
            ))
        ]
    return system_msgs + summary_injection + non_system_msgs

def get_thread_summary(thread_id: str) -> Optional[str]:
    return _THREAD_SUMMARIES.get(thread_id)

# ==============================================================================
# GRAPH NODES
# ==============================================================================

def _sanitize_tool_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    sanitized = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if not msg.content:
                msg = ToolMessage(content="Execution successful, but no output was returned.", tool_call_id=msg.tool_call_id, name=msg.name)
            elif not isinstance(msg.content, str):
                msg = ToolMessage(content=str(msg.content), tool_call_id=msg.tool_call_id, name=msg.name)
        sanitized.append(msg)
    return sanitized

def chat_llm(state: chat_state, config: RunnableConfig, store: BaseStore): 
    messages = state["messages"]
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    ns = ("user", user_id, "details")
    items = store.search(ns)
    user_details_content = "\n".join(f"- {it.value.get('data', '')}" for it in items) if items else "No known user details."

    managed_messages = manage_context(list(messages), thread_id)
    sanitized_messages = _sanitize_tool_messages(managed_messages)

    try:
        response = llm_with_tools.invoke([
            SystemMessage(content=(
                f"You are CaracalX Researcher, a helpful AI assistant with access to real-time tools. "
                f"The current chat thread_id is '{thread_id}'. "
                f"You must pass this exact thread_id whenever you use the rag_tool.\n\n"
                f"CRITICAL INSTRUCTION REGARDING PDFs: "
                f"You have a tool called 'rag_tool' specifically for searching and retrieving content from uploaded PDFs. "
                f"When a user asks about an attached PDF, document, or asks you to 'check the content', you MUST use the 'rag_tool'. "
                f"NEVER refuse by saying you cannot access local storage or files. The 'rag_tool' securely handles all file retrieval for you.\n\n"
                f"USER DETAILS (Long Term Memory):\n{user_details_content}\n\n"
                f"Personalize your responses based on the user details above when appropriate."
            )),
            *sanitized_messages
        ])
        return {"messages": [response]}
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"GRAPH LLM NODE ERROR: {repr(e)}")
        print("=" * 80 + "\n")
        raise

def remember_node(state: chat_state, config: RunnableConfig, store: BaseStore):
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    ns = ("user", user_id, "details")
    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage): return {}
    last_text = last_msg.content

    MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.
    CURRENT USER DETAILS (existing memories):
    {user_details_content}
    TASK:
    - Review the user's latest message.
    - Extract ONLY core identity, stable preferences, or ongoing projects/goals.
    - STRICT RULE: DO NOT extract transient conversational queries, questions, or specific tasks.
    - If the user's message is just a standard prompt or question, set should_write to False and return an empty list.
    - For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
    - Keep each memory as a short atomic sentence. No speculation; only facts stated by the user.
    
    OUTPUT FORMAT (JSON):
    {{
        "should_write": true/false,
        "memories": [
            {{"text": "memory text", "is_new": true}},
            {{"text": "another memory", "is_new": false}}
        ]
    }}
    """

    decision = None
    try:
        # We bypass memory_extractor (with_structured_output) entirely because it relies 
        # on LangChain's internal OutputParser, which often fails or returns None with custom wrappers.
        fallback_prompt = MEMORY_PROMPT.format(user_details_content=existing)
        response = memory_llm.invoke([
            SystemMessage(content=fallback_prompt + "\n\nYou MUST output valid JSON only. No markdown, no explanation, just raw JSON."),
            HumanMessage(content=last_text)
        ])
        
        # Safely extract text content
        if response and hasattr(response, "content") and response.content:
            text = response.content.strip()
            
            # Extract JSON from response (handles markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            decision = MemoryDecision(**data)
            
    except Exception as e:
        print(f"[Memory] JSON parsing failed: {e}. Defaulting to no memory.")
        
    # CRITICAL SAFETY NET: If decision is still None or invalid, default to no memory
    if not isinstance(decision, MemoryDecision):
        decision = MemoryDecision(should_write=False, memories=[])

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
    return {}
    
# ==============================================================================
# GRAPH COMPILATION
# ==============================================================================

toolnode = ToolNode(tools)
graph = StateGraph(chat_state)

graph.add_node('remember', remember_node)
graph.add_node('chat_node', chat_llm)
graph.add_node('tools', toolnode)

graph.add_edge(START, 'remember')
graph.add_edge('remember', 'chat_node') 
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=cheakpoint, store=store)

# ==============================================================================
# UI DATABASE HELPERS
# ==============================================================================

def get_threads():
    threads = {}
    cursor = conn.execute("SELECT thread_id, title FROM thread_metadata")
    for thread_id, title in cursor.fetchall():
        threads[thread_id] = title

    for checkpoint in cheakpoint.list(None):
        configurable = None
        try:
            configurable = checkpoint.config.get("configurable") if hasattr(checkpoint.config, "get") else getattr(checkpoint.config, "configurable", None)
        except Exception:
            configurable = None

        if not configurable: continue
        thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else getattr(configurable, "thread_id", None)
        if not thread_id: continue
        if thread_id not in threads:
            threads[thread_id] = "New chat"
    return threads

def save_thread_title(thread_id: str, title: str):
    conn.execute("INSERT OR REPLACE INTO thread_metadata(thread_id, title) VALUES (?, ?)", (thread_id, title))
    conn.commit()

def generate_title(message: str):
    prompt = f"Create a short chat title in under 5 words.\n\nMessage:\n{message}"
    response = model.invoke(prompt)
    return response.content.strip()

def delete_thread(thread_id: str):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    cursor.execute("DELETE FROM thread_metadata WHERE thread_id = ?", (thread_id,))
    conn.commit()
    
    _THREAD_SUMMARIES.pop(thread_id, None)
    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS