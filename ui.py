import streamlit as st
import os
import uuid
from src.engine import LocalRAGEngine
from src.exceptions import OffTopicException
from langchain_core.messages import HumanMessage, AIMessage

# 1. Page Configuration & Custom CSS Injection to look like Gemini
st.set_page_config(page_title="Gemini Clone", layout="wide")

st.markdown("""
    <style>
    /* Gemini-like styling */
    .stApp { background-color: #ffffff; color: #1f1f1f; font-family: 'Google Sans', sans-serif; }
    .stChatMessage { border-radius: 20px; margin-bottom: 15px; padding: 15px; border: none; }
    .stChatMessage.user { background-color: #f0f4f9; margin-left: auto; border-bottom-right-radius: 4px; max-width: 80%; }
    .stChatMessage.assistant { background-color: transparent; }
    [data-testid="stChatMessageContent"] { font-size: 16px; line-height: 1.5; }
    /* Hide the top header */
    header { visibility: hidden; }
    .file-chip { background-color: #e3e3e3; border-radius: 16px; padding: 6px 12px; display: inline-block; font-size: 0.85rem; margin-bottom: 8px; color: #1f1f1f; font-weight: 500; }
    
    /* Modify sidebar */
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: none; }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = f"tenant_{uuid.uuid4().hex}"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "uploaded_filenames" not in st.session_state:
    st.session_state["uploaded_filenames"] = set()

# Instantiate Engine safely with history_aware=True and persistent session_id
@st.cache_resource
def get_engine(session_id: str):
    return LocalRAGEngine(session_id=session_id, history_aware=True)

try:
    engine = get_engine(st.session_state.session_id)
except Exception as e:
    st.error(f"Could not connect to backend vector database or LLM services. Error: {e}")
    st.stop()

# Load already existing files from the engine's user_data_dir
if not st.session_state.get("files_loaded_initially"):
    if os.path.exists(engine.user_data_dir):
        for f in os.listdir(engine.user_data_dir):
            if os.path.isfile(os.path.join(engine.user_data_dir, f)):
                st.session_state["uploaded_filenames"].add(f)
    st.session_state["files_loaded_initially"] = True

# Sidebar for Chat Management
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        st.rerun()
        
    st.markdown("---")
    with st.sidebar.expander("📁 Manage Ingested Documents", expanded=True):
        if not st.session_state.get("uploaded_filenames"):
            st.caption("No context files loaded in current isolated workspace.")
        else:
            # Loop through active session elements and present targeted trash hooks
            for filename in list(st.session_state.uploaded_filenames):
                col1, col2 = st.columns([4, 1])
                
                # Truncate string gracefully for UI scannability
                display_name = filename if len(filename) <= 20 else f"{filename[:17]}..."
                col1.text(f"📄 {display_name}")
                
                # Wire up the button to invoke the backend deletion mechanism
                if col2.button("🗑️", key=f"del_{filename}"):
                    try:
                        # Invoke backend Qdrant vector wipe strategy
                        engine.delete_document(filename)
                        
                        # Evict from active state arrays tracking current page history
                        st.session_state.uploaded_filenames.remove(filename)
                        
                        st.sidebar.success(f"Removed reference vector space: {filename}")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Could not purge vector chunks: {str(e)}")

# Main Chat Interface
if not st.session_state["messages"]:
    st.markdown("<h1 style='text-align: center; color: #444746; margin-top: 5vh; font-size: 3rem;'>Hello</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #c4c7c5; margin-bottom: 10vh;'>How can I help you today?</h2>", unsafe_allow_html=True)
else:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                if msg.get("files"):
                    for f in msg["files"]:
                        st.markdown(f"<div class='file-chip'>📎 {f}</div>", unsafe_allow_html=True)
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("Sources"):
                        for i, doc_text in enumerate(msg["sources"][:3]):
                            st.markdown(f"**Source {i+1}**")
                            st.info(doc_text[:300] + "...")

# Bottom Command Interface Tray
uploaded_files = st.file_uploader(
    "Upload context files (PDF, TXT, DOCX, MD, HTML)", 
    accept_multiple_files=True, 
    label_visibility="collapsed"
)

prompt_data = st.chat_input("Ask me anything...")

if prompt_data:
    active_filenames = []
    
    # Process files if staged
    if uploaded_files:
        progress_bar = st.progress(0, text="Starting ingestion...")
        total_files = len(uploaded_files)
        
        for idx, f in enumerate(uploaded_files):
            active_filenames.append(f.name)
            st.session_state["uploaded_filenames"].add(f.name)
            progress_bar.progress((idx) / total_files, text=f"Ingesting {f.name}...")
            try:
                file_bytes = f.read()
                engine.ingest_single_document(f.name, file_bytes)
            except Exception as e:
                st.error(f"Failed to ingest file {f.name}: {str(e)}")
                st.stop()
                
        progress_bar.progress(1.0, text="Ingestion complete!")
                
    st.session_state["messages"].append({
        "role": "user",
        "content": prompt_data,
        "files": active_filenames
    })
    
    st.rerun()

# Execute model text inference streaming if user just updated the message timeline
if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    last_prompt = st.session_state["messages"][-1]["content"]
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        context_docs = []
        
        try:
            inputs = {
                "input": last_prompt,
                "chat_history": st.session_state["chat_history"]
            }
            
            with st.spinner("Thinking..."):
                for chunk in engine.stream(inputs):
                    if "context" in chunk:
                        context_docs.extend(chunk["context"])
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            sources = []
            if context_docs:
                with st.expander("Sources"):
                    for i, doc in enumerate(context_docs[:3]):
                        st.markdown(f"**Source {i+1}**")
                        st.info(doc.page_content[:300] + "...")
                        sources.append(doc.page_content)
                        
            st.session_state["messages"].append({"role": "assistant", "content": full_response, "sources": sources})
            
            # Update chat history
            st.session_state["chat_history"].extend([
                HumanMessage(content=last_prompt),
                AIMessage(content=full_response)
            ])
            
        except OffTopicException:
            placeholder.warning("This query is outside the scope of your ingested document context bounds.")
            st.session_state["messages"].append({"role": "assistant", "content": "This query is outside the scope of your ingested document context bounds."})
            st.session_state["chat_history"].extend([
                HumanMessage(content=last_prompt),
                AIMessage(content="This query is outside the scope of your ingested document context bounds.")
            ])
        except Exception as e:
            placeholder.error(f"Error: {str(e)}")