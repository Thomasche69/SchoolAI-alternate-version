import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import pdfplumber
import re

st.markdown("""
<style>
    /* Your existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
from datetime import datetime
MODEL_CACHE = {}
EMBEDDING_CACHE = {}

def get_llm(model_name):
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = ChatOllama(model=model_name)
    return MODEL_CACHE[model_name]

def get_embedding():
    if "qwen3-embedding:0.6b" not in EMBEDDING_CACHE:
        EMBEDDING_CACHE["qwen3-embedding:0.6b"] = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    return EMBEDDING_CACHE["qwen3-embedding:0.6b"]

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = "general"

if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embedding=get_embedding())



current_date = datetime.now().strftime("%B %d, %Y")
# System prompts for different modes
pdf_system_prompt = SystemMessagePromptTemplate.from_template(
   """You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know."""
)

general_system_prompt = SystemMessagePromptTemplate.from_template(
   """You are a helpful and knowledgeable AI assistant. Provide clear, accurate, and thoughtful responses to user queries. 
Be conversational and helpful."""
)

 
# UI Elements
st.title("AI Assistant")
st.markdown("### Your Intelligent Research Assistant")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode selection
    st.subheader("Chat Mode")
    mode = st.radio(
        "Choose a mode:",
        ["general", "pdf"],
        format_func=lambda x: "General Chat" if x == "general" else "PDF Chat",
    )
    st.session_state.chat_mode = mode
    
    st.divider()
    
    selected_model = st.selectbox(
        "Choose Model",
        ["qwen3:8b", "llama3:8b"],
        index=0
    )
    
    st.divider()
    st.markdown("### Model Capabilities")
    if mode == "pdf":
        st.markdown("""
        - PDF assistant
        - Helps summarize PDFs
        - Answer questions about your documents
        """)
    else:
        st.markdown("""
        - General conversation
        - Answer any questions
        - Provide information and assistance
        """)

PDF_STORAGE_PATH = 'document_store/'
MAX_PDF_PAGES = 100
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(raw_documents)

def index_documents(document_chunks):
    try:
        texts = [doc.page_content for doc in document_chunks]
        metadatas = [doc.metadata for doc in document_chunks]
        st.session_state.vector_store.add_texts(texts, metadatas)
    except Exception as e:
        st.error(f"Error adding documents to vector store: {e}")

def find_related_documents(query):
    try:
        return st.session_state.vector_store.similarity_search(query)
    except Exception as e:
        st.error(f"Error searching vector store: {e}")
        return []

# PDF Upload section
if st.session_state.chat_mode == "pdf":
    st.subheader("📄 Document Upload")
    uploaded_pdf = st.file_uploader(
        "Upload Research Document (PDF)",
        type="pdf",
        help="Select a PDF document to analyze",
        accept_multiple_files=False
    )
else:
    uploaded_pdf = None

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    with pdfplumber.open(saved_path) as pdf:
        num_pages = len(pdf.pages)
    if num_pages > MAX_PDF_PAGES:
        st.error(f"❌ PDF has {num_pages} pages. Please upload a PDF with {MAX_PDF_PAGES} pages or fewer.")
        os.remove(saved_path)
    else:
        with st.spinner("Processing PDF..."):
            raw_docs = load_pdf_documents(saved_path)
            processed_chunks = chunk_documents(raw_docs)
            index_documents(processed_chunks)
            st.session_state.pdf_uploaded = True
        st.success("✅ Document processed successfully! Ask your questions below.")

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your AI assistant. How can I help you today?"}]

def render_response_with_math_and_thinking(content):
    pattern = re.compile(
        r"(<think>.*?</think>)|"
        r"(\\\((.*?)\\\))|"
        r"(\\\[(.*?)\\\])",
        re.DOTALL
    )

    pos = 0
    for match in pattern.finditer(content):
        start, end = match.span()
        if start > pos:
            st.markdown(content[pos:start], unsafe_allow_html=True)
        if match.group(1):
            inner = match.group(1)[7:-8]
            st.markdown(
                f'<span style="color:#bbbbbb;font-style:italic;">{inner}</span>',
                unsafe_allow_html=True
            )
        elif match.group(2):
            st.latex(match.group(3).strip())
        elif match.group(4):
            st.latex(match.group(5).strip())
        pos = end
    if pos < len(content):
        st.markdown(content[pos:], unsafe_allow_html=True)

def build_prompt_chain(user_query, context_documents=None):
    mode = st.session_state.chat_mode
    
    if mode == "pdf":
        system_prompt = pdf_system_prompt
        context = "\n\n".join([doc.page_content for doc in context_documents]) if context_documents else ""
    else:
        system_prompt = general_system_prompt
        context = ""
    
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(content))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(content))
    
    escaped_query = user_query.replace("{", "{{").replace("}", "}}")
    
    if mode == "pdf" and context:
        escaped_context = context.replace("{", "{{").replace("}", "}}")
        prompt_sequence.append(
            HumanMessagePromptTemplate.from_template(
                f"Context:\n{escaped_context}\n\nQuery: {escaped_query}"
            )
        )
    else:
        prompt_sequence.append(
            HumanMessagePromptTemplate.from_template(escaped_query)
        )
    
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response_streaming(prompt_chain, selected_model, message_placeholder):
    try:
        prompt_value = prompt_chain.format_prompt()
        # Pass message objects directly instead of formatting to string
        input_for_model = prompt_value.to_messages()
    except Exception as e:
        # Fallback: ChatOllama can accept a plain string
        input_for_model = str(prompt_chain)
        st.warning(f"Prompt formatting failed: {e}")

    ollama_model = get_llm(selected_model)
    full_response = ""
    
    # Stream with messages (or fallback string)
    for chunk in ollama_model.stream(input_for_model):
        # Extract content from AIMessageChunk objects
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_response += content
        
        with message_placeholder.container():
            render_response_with_math_and_thinking(full_response + "▌")

    with message_placeholder.container():
        render_response_with_math_and_thinking(full_response)

    return full_response

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            if message["role"] == "ai":
                render_response_with_math_and_thinking(message["content"])
            else:
                st.markdown(message["content"])

# Chat input
if st.session_state.chat_mode == "pdf":
    placeholder_text = "Enter your question about the document..."
else:
    placeholder_text = "Enter your question..."

user_input = st.chat_input(placeholder_text)

if user_input:
    if st.session_state.chat_mode == "pdf" and not st.session_state.pdf_uploaded:
        st.error("❌ Please upload and process a PDF before asking questions.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if st.session_state.chat_mode == "pdf":
            relevant_docs = find_related_documents(user_input)
            prompt_chain = build_prompt_chain(user_input, relevant_docs)
        else:
            prompt_chain = build_prompt_chain(user_input)
        
        with st.chat_message("ai"):
            message_placeholder = st.empty()
            ai_response = generate_ai_response_streaming(prompt_chain, selected_model, message_placeholder)

        st.session_state.message_log.append({"role": "user", "content": user_input})
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
# ...existing code...
