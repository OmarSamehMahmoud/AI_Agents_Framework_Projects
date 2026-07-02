# ==============================================================================
# LIBRARIES AND DEPENDENCIES IMPORTS
# ==============================================================================
import streamlit as st  # Core framework for building the interactive web UI
import uuid  # Standard library for generating unique random IDs for chat sessions
from dotenv import load_dotenv  # Utility to load environment variables from a .env file securely

# Backend objects are initialized during import, so environment variables must exist first.
load_dotenv()

# Import the compiled LangGraph agent, database helpers, and context management variables from the backend
from backend import (
    chatbot, get_threads, save_thread_title, generate_title,
    delete_thread, thread_document_metadata, thread_has_document,
    ingest_pdf, get_thread_summary,
    RECENT_MESSAGES_TO_KEEP, SUMMARIZATION_TRIGGER, HARD_TRIM_LIMIT
)
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk  # LangChain schemas for structuring chat data

# Configure the fundamental properties of the Streamlit webpage
st.set_page_config(
    page_title="CaracalX Researcher",
    page_icon="🐆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS / HTML STYLING (PREMIUM DARK GLASSMORPHISM THEME)
# ==============================================================================
st.markdown(
    """
    <style>
        /* ── CORE VARIABLES ── */
        :root {
            --bg-deep: #050608;
            --bg-surface: #0f1117;
            --bg-glass: rgba(15, 17, 23, 0.75);
            --text-main: #e6e8eb;
            --text-muted: #8a93a6;
            --accent-caracal: #ff7b00;
            --accent-cyan: #00e5ff;
            --line-subtle: rgba(255, 255, 255, 0.06);
        }

        /* ── GLOBAL APP OVERRIDE (Force Dark Mode) ── */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: var(--bg-deep) !important;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(255, 123, 0, 0.06) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(0, 229, 255, 0.04) 0%, transparent 45%),
                linear-gradient(180deg, #050608 0%, #0a0c12 100%) !important;
            color: var(--text-main) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Hide default Streamlit header/footer */
        header[data-testid="stHeader"] { background: transparent !important; }
        footer { display: none !important; }
        
        /* ── SIDEBAR GLASSMORPHISM ── */
        section[data-testid="stSidebar"] {
            background: rgba(10, 12, 18, 0.85) !important;
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-right: 1px solid var(--line-subtle) !important;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 2rem;
        }

        /* ── BRANDING ── */
        .caracal-brand {
            display: flex; align-items: center; gap: 1rem;
            margin-bottom: 2rem; padding: 0 0.5rem;
        }
        .caracal-mark {
            width: 42px; height: 42px;
            display: grid; place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg, #ff7b00 0%, #ff4500 100%);
            box-shadow: 0 0 20px rgba(255, 123, 0, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.2);
            color: white; font-weight: 800; font-size: 1.4rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .caracal-brand-copy strong {
            display: block; font-size: 1.2rem; color: #fff;
            letter-spacing: 0.05em; text-transform: uppercase;
        }
        .caracal-brand-copy span {
            font-size: 0.75rem; color: var(--accent-cyan);
            letter-spacing: 0.1em; text-transform: uppercase;
        }

        /* ── SIDEBAR BUTTONS & INTERACTIONS ── */
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid var(--line-subtle) !important;
            color: #d1d5db !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 123, 0, 0.1) !important;
            border-color: var(--accent-caracal) !important;
            color: #fff !important;
            box-shadow: 0 0 15px rgba(255, 123, 0, 0.15) !important;
        }
        
        /* Primary New Chat Button */
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, rgba(255,123,0,0.2), rgba(255,123,0,0.05)) !important;
            border: 1px solid rgba(255, 123, 0, 0.4) !important;
            color: var(--accent-caracal) !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, rgba(255,123,0,0.3), rgba(255,123,0,0.1)) !important;
            box-shadow: 0 0 25px rgba(255, 123, 0, 0.3) !important;
        }

        /* File Uploader */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px dashed rgba(0, 229, 255, 0.3) !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-cyan) !important;
            background: rgba(0, 229, 255, 0.05) !important;
        }

        /* ── MAIN CONTENT AREA ── */
        [data-testid="stMainBlockContainer"] {
            max-width: 1000px;
            padding-top: 2rem;
            padding-bottom: 6rem;
        }

        /* ── HERO BANNER ── */
        .caracal-hero {
            position: relative;
            padding: 2.5rem;
            margin-bottom: 3rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255, 123, 0, 0.04) 0%, rgba(0, 229, 255, 0.02) 100%);
            border: 1px solid var(--line-subtle);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
            overflow: hidden;
        }
        .caracal-hero::before {
            content: ''; position: absolute; top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(255,123,0,0.1) 0%, transparent 60%);
            animation: pulse 8s infinite alternate;
        }
        @keyframes pulse {
            0% { transform: translate(0, 0); }
            100% { transform: translate(10%, 10%); }
        }
        .caracal-hero-content { position: relative; z-index: 1; }
        .caracal-eyebrow {
            color: var(--accent-cyan); font-size: 0.8rem; font-weight: 700;
            letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.5rem;
        }
        .caracal-hero h1 {
            margin: 0; font-size: 3.2rem; font-weight: 800;
            background: linear-gradient(90deg, #fff 0%, #ff7b00 50%, #00e5ff 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            line-height: 1.1; letter-spacing: -0.03em;
        }
        .caracal-hero p {
            color: #a0aabf; font-size: 1.1rem; max-width: 600px; margin-top: 1rem;
        }

        /* Stats Pills */
        .caracal-meta { display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap; }
        .caracal-stat {
            padding: 0.6rem 1.2rem; border-radius: 99px;
            background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.2);
            color: #fff; font-size: 0.85rem; font-weight: 600;
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.1);
        }
        .caracal-status {
            display: flex; align-items: center; gap: 0.6rem;
            padding: 0.6rem 1.2rem; border-radius: 99px;
            background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3);
            color: #34d399; font-weight: 600; font-size: 0.85rem;
        }
        .caracal-status-dot {
            width: 8px; height: 8px; border-radius: 50%; background: #34d399;
            box-shadow: 0 0 10px #34d399; animation: blink 2s infinite;
        }
        @keyframes blink { 50% { opacity: 0.4; } }

        /* ── EMPTY STATE ── */
        .caracal-empty {
            text-align: center; padding: 4rem 2rem;
            border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1);
            background: rgba(255,255,255,0.01);
        }
        .caracal-empty-icon {
            font-size: 3rem; margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #ff7b00, #00e5ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .suggestion-row { display: flex; gap: 1rem; justify-content: center; margin-top: 2rem; flex-wrap: wrap; }
        .suggestion-chip {
            padding: 0.8rem 1.5rem; border-radius: 12px;
            background: rgba(255,255,255,0.04); border: 1px solid var(--line-subtle);
            color: #cbd5e1; font-weight: 500; transition: all 0.2s;
            cursor: default;
        }
        .suggestion-chip:hover {
            background: rgba(255, 123, 0, 0.1); border-color: var(--accent-caracal);
            color: #fff; transform: translateY(-2px);
        }

        /* ── CHAT BUBBLES ── */
        [data-testid="stChatMessage"] {
            background: transparent !important; 
            border: none !important;
            box-shadow: none !important; 
            padding: 0 !important; 
            margin-bottom: 1.5rem !important;
        }
        
        /* User Bubble */
        [data-testid="stChatMessage"][data-chatmessage-role="user"] {
            background: linear-gradient(135deg, rgba(255, 123, 0, 0.12) 0%, rgba(255, 123, 0, 0.04) 100%) !important;
            border: 1px solid rgba(255, 123, 0, 0.25) !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 1.2rem 1.6rem !important;
            box-shadow: 0 8px 24px rgba(255, 123, 0, 0.05) !important;
            color: #f8fafc !important;
        }
        
        /* AI Bubble */
        [data-testid="stChatMessage"][data-chatmessage-role="assistant"] {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%) !important;
            border: 1px solid rgba(0, 229, 255, 0.15) !important;
            border-radius: 18px 18px 18px 4px !important;
            padding: 1.2rem 1.6rem !important;
            box-shadow: 0 8px 24px rgba(0, 229, 255, 0.03) !important;
            color: #e2e8f0 !important;
        }

        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, 
        [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, 
        [data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] code {
            color: inherit !important;
        }
        
        [data-testid="stChatMessage"] pre {
            background: #0d0f14 !important;
            border: 1px solid var(--line-subtle) !important;
            border-radius: 10px !important;
        }

        /* ── CHAT INPUT BAR ── */
        [data-testid="stBottomBlockContainer"] {
            background: transparent !important;
        }
        [data-testid="stChatInput"] {
            background: rgba(15, 17, 23, 0.8) !important;
            border: 1px solid rgba(255, 123, 0, 0.3) !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
            backdrop-filter: blur(10px);
        }
        [data-testid="stChatInput"] textarea {
            color: #fff !important;
            background: transparent !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }
        
        /* Send Button */
        [data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #ff7b00, #ff4500) !important;
            color: white !important;
            box-shadow: 0 0 20px rgba(255, 123, 0, 0.4) !important;
            border-radius: 10px !important;
        }

        /* ── SCROLLBARS ── */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { 
            background: rgba(255, 255, 255, 0.1); border-radius: 4px; 
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255, 123, 0, 0.4); }

    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE & THREAD MANAGEMENT UTILITIES
# ==============================================================================

def gen_thread():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = gen_thread()
    st.session_state.message_history = []
    st.session_state.thread_id = thread_id # type: ignore
    add_thread(thread_id)

def add_thread(thread_id, label="New chat"):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads[thread_id] = label
        save_thread_title(thread_id, label)

def load_thread(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": st.session_state.user_id 
        }
    }
    state = chatbot.get_state(config=config) # type: ignore
    return state.values.get("messages", [])

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_1" 

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = get_threads()

if 'thread_id' not in st.session_state:
    new_thread = gen_thread()
    st.session_state.thread_id = new_thread
    st.session_state.chat_threads[new_thread] = 'New chat'

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state.thread_id)

# ==============================================================================
# SIDEBAR UI
# ==============================================================================

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

st.sidebar.markdown(
    """
    <div class="caracal-brand">
        <div class="caracal-mark">C</div>
        <div class="caracal-brand-copy">
            <strong>CaracalX</strong>
            <span>Researcher</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div style="display: inline-flex; align-items: center; gap: 0.4rem; margin-bottom: 1.5rem; padding: 0.4rem 0.8rem; border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 999px; background: rgba(0, 229, 255, 0.05); color: #00e5ff; font-size: 0.75rem; font-weight: 600;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #00e5ff; box-shadow: 0 0 8px #00e5ff;"></span>
        Session {thread_key[:8]}
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button(
    "New Hunt",
    icon=":material/add_comment:",
    use_container_width=True,
    type="primary",
):
    reset_chat()
    st.rerun()

st.sidebar.markdown("#### Knowledge Base")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"**{latest_doc.get('filename')}**\n\n"
        f"{latest_doc.get('documents')} pages | {latest_doc.get('chunks')} chunks",
        icon=":material/check_circle:",
    )
else:
    st.sidebar.info(
        "Upload a PDF to ground answers in your own data.",
        icon=":material/upload_file:",
    )

uploaded_pdf = st.sidebar.file_uploader(
    "Add a source",
    type=["pdf"],
    help="Text-based PDFs work best.",
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
            except ValueError as exc:
                status_box.update(label="PDF could not be indexed", state="error", expanded=True)
                st.sidebar.error(str(exc))
            except Exception as exc:
                status_box.update(label="PDF indexing failed", state="error", expanded=True)
                st.sidebar.error(f"Could not process this PDF: {exc}")
            else:
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="PDF ready", state="complete", expanded=False)

st.sidebar.divider()
st.sidebar.markdown("#### Memory Core")

thread_summary = get_thread_summary(thread_key)

if thread_summary:
    with st.sidebar.expander(
        "View active memory",
        expanded=False,
        icon=":material/notes:",
    ):
        st.caption(thread_summary)
    st.sidebar.caption(
        f"Keeping last **{RECENT_MESSAGES_TO_KEEP}** messages. "
        f"Compression at **{SUMMARIZATION_TRIGGER}**."
    )
else:
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_key, 
                "user_id": st.session_state.user_id
            }
        }
    )
    current_count = len(state.values.get("messages", []))
    remaining = max(0, SUMMARIZATION_TRIGGER - current_count)

    if remaining > 0:
        st.sidebar.caption(
            f"No summary yet. Compression in "
            f"~**{remaining}** message(s)."
        )
    else:
        st.sidebar.caption("Summary generation on next response.")

st.sidebar.divider()
st.sidebar.markdown("#### Recent Hunts")

for thread_id, label in reversed(list(st.session_state.chat_threads.items())):
    col1, col2 = st.sidebar.columns([5, 1])

    with col1:
        active_prefix = "🐾 " if str(thread_id) == thread_key else ""
        if st.button(
            f"{active_prefix}{label}",
            key=f"load_{thread_id}",
            use_container_width=True,
        ):
            st.session_state.thread_id = thread_id
            messages = load_thread(thread_id)
            
            temp_msg = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content
                
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "") 
                        for block in content 
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    content = "".join(text_parts)
                elif not isinstance(content, str):
                    content = str(content)
                    
                temp_msg.append({"role": role, "content": content})
            
            st.session_state.message_history = temp_msg

    with col2:
        if st.button(
            "",
            key=f"delete_{thread_id}",
            icon=":material/delete:",
            help=f"Delete {label}",
            use_container_width=True,
        ):
            delete_thread(thread_id)
            st.session_state.chat_threads.pop(thread_id, None)
            
            if st.session_state.thread_id == thread_id:
                new_thread = gen_thread()
                st.session_state.thread_id = new_thread
                st.session_state.message_history = []
                st.session_state.chat_threads[new_thread] = "New Hunt"
            
            st.rerun()

# ==============================================================================
# MAIN CHAT INTERFACE
# ==============================================================================

message_count = len(st.session_state.message_history)
source_count = len(thread_docs)

st.markdown(
    f"""
    <div class="caracal-hero">
        <div class="caracal-hero-content">
            <div class="caracal-eyebrow">CaracalX Intelligence</div>
            <h1>Uncover. Analyze. Master.</h1>
            <p>Deploy CaracalX to hunt down information, synthesize complex documents, and execute advanced reasoning tasks.</p>
            <div class="caracal-meta">
                <div class="caracal-stat">{message_count} interactions</div>
                <div class="caracal-stat">{source_count} PDF source{"s" if source_count != 1 else ""}</div>
                <div class="caracal-status">
                    <span class="caracal-status-dot"></span>
                    Neural Core Online
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.message_history:
    st.markdown(
        """
        <div class="caracal-empty">
            <div class="caracal-empty-icon">🐾</div>
            <h2 style="color: #fff; font-size: 1.8rem; margin-bottom: 0.5rem;">What are we hunting today?</h2>
            <p style="color: #94a3b8; max-width: 500px; margin: 0 auto;">
                Deploy CaracalX to analyze your PDFs, scrape the live web, or execute complex multi-step reasoning chains.
            </p>
            <div class="suggestion-row">
                <span class="suggestion-chip">🔍 Deep Web Research</span>
                <span class="suggestion-chip">📄 Synthesize my PDF</span>
                <span class="suggestion-chip">🧠 Complex Logic Chain</span>
                <span class="suggestion-chip">📈 Market Analysis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.message_history:
    avatar = "🧑‍💻" if message["role"] == "user" else "🐆"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

user_input = st.chat_input("Message CaracalX...")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "user_id": st.session_state.user_id 
        }
    }

    if st.session_state.chat_threads[st.session_state.thread_id] == "New chat":
        title = generate_title(user_input)
        if len(user_input) > 30:
            title += "..."
        st.session_state.chat_threads[st.session_state.thread_id] = title
        save_thread_title(st.session_state.thread_id, title)

    with st.chat_message("assistant", avatar="🐆"):
        
        progress = st.empty()
        progress.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.8rem; padding: 0.8rem 1.2rem; background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 12px; color: #00e5ff; font-weight: 600; font-size: 0.9rem; margin-bottom: 1rem;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #00e5ff; box-shadow: 0 0 10px #00e5ff; animation: blink 1s infinite;"></span>
                CaracalX is processing...
            </div>
            """,
            unsafe_allow_html=True,
        )
        executed_tools = []

        def ai_onlystream():
            try:
                # Use .invoke() (non-streaming) for reliability, then simulate streaming in the UI
                full_response = chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                
                # Extract the final AI message content
                ai_message = None
                if isinstance(full_response, dict) and "messages" in full_response:
                    messages = full_response["messages"]
                    if messages:
                        ai_message = messages[-1]
                
                if ai_message and hasattr(ai_message, 'content') and ai_message.content:
                    content = ai_message.content
                    # Simulate streaming by yielding one character at a time
                    for char in content:
                        yield char
                else:
                    yield "⚠️ No response generated. The model may have failed silently."

            except Exception as e:
                progress.empty()
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                yield error_msg
                
        ai_msg = st.write_stream(ai_onlystream())
        progress.empty()

        # SAFETY NET: If the model returned completely empty output, show a warning instead of a blank bubble
        if not ai_msg:
            ai_msg = "⚠️ The model returned an empty response. (Check your terminal for Ollama errors or debug logs)."
            st.warning(ai_msg)

    st.session_state.message_history.append({"role": "assistant", "content": ai_msg})