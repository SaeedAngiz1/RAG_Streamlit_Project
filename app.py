"""
RAG Q&A System with AI Memory & Multiple AI Providers
Supports: OpenAI, Ollama, Anthropic, Custom API
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# Memory directory
MEMORY_DIR = "memory_files"
os.makedirs(MEMORY_DIR, exist_ok=True)

# ============================================
# CONFIGURATION
# ============================================
def get_config() -> Dict[str, Any]:
    """Get or create configuration"""
    defaults = {
        "provider": "openai",
        "openai": {
            "api_key": "",
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com/v1/chat/completions"
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llama3.2"
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-3-5-haiku-20241007",
            "base_url": "https://api.anthropic.com/v1/messages"
        },
        "custom": {
            "api_key": "",
            "base_url": "",
            "model": "",
            "payload_template": '{"model":"{model}","messages":[{"role":"user","content":"{prompt}"}]}'
        }
    }
    
    if "ai_config" not in st.session_state:
        st.session_state.ai_config = defaults
    
    return st.session_state.ai_config

def save_config(config: Dict[str, Any]):
    """Save configuration"""
    st.session_state.ai_config = config

# ============================================
# AI PROVIDER CONNECTIONS
# ============================================
def get_llm_from_config(config: Dict[str, Any]):
    """Get LLM based on current configuration"""
    from langchain.chat_models import ChatOpenAI
    from langchain.chat_models import ChatOllama
    from langchain.schema import HumanMessage
    
    provider = config.get("provider", "openai")
    
    if provider == "openai":
        p = config.get("openai", {})
        return ChatOpenAI(
            model_name=p.get("model", "gpt-3.5-turbo"),
            api_key=p.get("api_key", ""),
            temperature=0.3,
            streaming=True
        )
    
    elif provider == "ollama":
        p = config.get("ollama", {})
        return ChatOllama(
            base_url=p.get("base_url", "http://localhost:11434"),
            model=p.get("model", "llama3.2"),
            temperature=0.3,
            stream=True
        )
    
    elif provider == "anthropic":
        # Anthropic requires special handling
        p = config.get("anthropic", {})
        return AnthropicLLM(
            api_key=p.get("api_key", ""),
            model=p.get("model", "claude-3-5-haiku-20241007"),
            base_url=p.get("base_url", "https://api.anthropic.com/v1/messages")
        )
    
    elif provider == "custom":
        p = config.get("custom", {})
        return CustomLLM(
            api_key=p.get("api_key", ""),
            base_url=p.get("base_url", ""),
            model=p.get("model", ""),
            payload_template=p.get("payload_template", '{"model":"{model}","messages":[{"role":"user","content":"{prompt}"}]}')
        )
    
    # Default to OpenAI
    return ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

# ============================================
# CUSTOM LLM CLASSES
# ============================================
class AnthropicLLM:
    """Custom Anthropic LLM wrapper"""
    
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    def __call__(self, messages: List[Dict], **kwargs):
        from langchain.schema import AIMessage, HumanMessage
        
        # Convert messages format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": anthropic_messages
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        return AIMessage(content=data["content"][0]["text"])

class CustomLLM:
    """Custom API LLM wrapper"""
    
    def __init__(self, api_key: str, base_url: str, model: str, payload_template: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.payload_template = payload_template
    
    def __call__(self, messages: List[Dict], **kwargs):
        from langchain.schema import AIMessage
        
        # Get the last user message
        user_prompt = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        
        # Build payload
        rendered = self.payload_template.replace("{prompt}", user_prompt.replace('"', '\\"'))
        rendered = rendered.replace("{model}", self.model)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(self.base_url, headers=headers, json=json.loads(rendered), timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Try different response formats
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
        elif "response" in data:
            content = data["response"]
        elif "message" in data:
            content = data["message"].get("content", str(data))
        else:
            content = str(data)
        
        return AIMessage(content=content)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "vectorstore": None,
        "retriever": None,
        "qa_chain": None,
        "documents": [],
        "chunks": [],
        "messages": [],
        "conversation_id": str(uuid.uuid4())[:8],
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "username": "",
        "session_start": datetime.now(),
        "conversation_blueprint": {
            "main_topic": "General Q&A",
            "user_intent": [],
            "key_entities": [],
            "conversation_flow": [],
            "total_questions": 0
        },
        "loaded_memories": [],
        "document_stats": {
            "total_docs": 0,
            "total_chunks": 0,
            "sources": []
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize config
    get_config()

# ============================================
# DOCUMENT PROCESSING
# ============================================
def process_documents(uploaded_files: List, embeddings_model: str = "all-MiniLM-L6-v2") -> bool:
    """Process uploaded documents and create RAG pipeline"""
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    try:
        all_text = ""
        
        for file in uploaded_files:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                all_text += f"\n\n--- {file.name} ---\n" + df.to_string()
            elif file.name.endswith('.txt'):
                content = file.read().decode('utf-8')
                all_text += f"\n\n--- {file.name} ---\n" + content
            elif file.name.endswith('.md'):
                content = file.read().decode('utf-8')
                all_text += f"\n\n--- {file.name} ---\n" + content
            else:
                content = file.read().decode('utf-8', errors='ignore')
                all_text += f"\n\n--- {file.name} ---\n" + content
        
        doc = Document(page_content=all_text, metadata={"source": "uploaded_documents"})
        st.session_state.documents = [doc]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(st.session_state.documents)
        st.session_state.chunks = chunks
        
        st.session_state.document_stats = {
            "total_docs": len(uploaded_files),
            "total_chunks": len(chunks),
            "sources": [f.name for f in uploaded_files]
        }
        
        with st.spinner("Creating embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Get LLM based on config
            config = get_config()
            llm = get_llm_from_config(config)
            
            template = """You are a helpful AI assistant. Use the following context to answer the question.
If the context doesn't contain relevant information, say you don't know.

Context: {context}
Question: {question}

Answer in a clear, structured format."""

            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.qa_chain = qa_chain
        
        return True
    
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return False

# ============================================
# CONVERSATION BLUEPRINT
# ============================================
def update_blueprint(user_message: str):
    """Update conversation blueprint"""
    blueprint = st.session_state.conversation_blueprint
    
    keywords = [word for word in user_message.split() 
                if len(word) > 4 and word.lower() not in 
                ['what', 'which', 'where', 'when', 'how', 'explain', 'describe']]
    
    blueprint["key_entities"] = list(set(blueprint["key_entities"] + keywords[:5]))
    blueprint["total_questions"] += 1
    
    blueprint["conversation_flow"].append({
        "timestamp": datetime.now().isoformat(),
        "question_preview": user_message[:50] + "..." if len(user_message) > 50 else user_message
    })
    
    if "?" in user_message:
        if "what" in user_message.lower():
            blueprint["user_intent"].append("definition")
        elif "how" in user_message.lower():
            blueprint["user_intent"].append("procedure")
        elif "why" in user_message.lower():
            blueprint["user_intent"].append("explanation")

# ============================================
# MEMORY SYSTEM
# ============================================
def generate_memory_file() -> str:
    """Generate memory file content"""
    blueprint = st.session_state.conversation_blueprint
    session_duration = datetime.now() - st.session_state.session_start
    config = get_config()
    
    memory_content = f"""# AI Session Memory

## User Information
- **Username:** {st.session_state.username or "Anonymous"}
- **Session ID:** {st.session_state.conversation_id}
- **Session Start:** {st.session_state.session_start.strftime('%Y-%m-%d %H:%M:%S')}
- **Session End:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Duration:** {str(session_duration).split('.')[0]}
- **AI Provider:** {config.get('provider', 'unknown').upper()}

## Token Usage
- **Input Tokens:** {st.session_state.input_tokens:,}
- **Output Tokens:** {st.session_state.output_tokens:,}
- **Total Tokens:** {st.session_state.total_tokens:,}
- **Estimated Cost:** ${st.session_state.total_tokens * 0.000002:.4f}

## Conversation Blueprint
- **Main Topic:** {blueprint['main_topic']}
- **Total Questions:** {blueprint['total_questions']}
- **Question Types:** {', '.join(set(blueprint['user_intent'])) if blueprint['user_intent'] else 'General'}

### Key Entities
{chr(10).join(f'- {entity}' for entity in blueprint['key_entities'][:20])}

### Conversation Flow
"""
    
    for i, flow in enumerate(blueprint['conversation_flow'], 1):
        memory_content += f"| {i} | {flow['timestamp']} | {flow['question_preview']} |\n"
    
    memory_content += f"""
## Document Sources
- **Documents:** {st.session_state.document_stats['total_docs']}
- **Chunks:** {st.session_state.document_stats['total_chunks']}

## Chat History
"""
    
    for msg in st.session_state.messages[-20:]:
        role = "User" if msg["role"] == "user" else "AI"
        memory_content += f"\n### {role}\n{msg['content']}\n"
    
    memory_content += "\n\n---\n*Generated by RAG Streamlit App*\n"
    
    return memory_content

def save_memory_to_file():
    """Save memory to file"""
    memory_content = generate_memory_file()
    filename = f"memory_{st.session_state.username or 'user'}_{st.session_state.conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(MEMORY_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(memory_content)
    
    return filepath, filename

def load_memory_file(file) -> Dict[str, Any]:
    """Load and parse memory file"""
    try:
        content = file.read().decode('utf-8')
        
        memory_data = {
            "content": content,
            "username": "",
            "session_id": "",
            "total_tokens": 0,
            "provider": ""
        }
        
        lines = content.split('\n')
        for line in lines:
            if '**Username:**' in line:
                memory_data['username'] = line.split('**Username:**')[1].strip()
            elif '**Session ID:**' in line:
                memory_data['session_id'] = line.split('**Session ID:**')[1].strip()
            elif '**Total Tokens:**' in line:
                try:
                    memory_data['total_tokens'] = int(line.split('**Total Tokens:**')[1].strip().replace(',', ''))
                except:
                    pass
            elif '**AI Provider:**' in line:
                memory_data['provider'] = line.split('**AI Provider:**')[1].strip()
        
        return memory_data
    
    except Exception as e:
        return {"error": str(e)}

# ============================================
# SIDEBAR: AI MEMORY
# ============================================
def render_sidebar_memory():
    """Render AI Memory section"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("## 🧠 AI Memory")
        
        st.session_state.username = st.text_input(
            "👤 Username",
            value=st.session_state.username,
            placeholder="Enter your name"
        )
        
        session_duration = datetime.now() - st.session_state.session_start
        st.markdown(f"""
        **📅 Session Info**
        - Started: {st.session_state.session_start.strftime('%H:%M:%S')}
        - Duration: {str(session_duration).split('.')[0]}
        - ID: `{st.session_state.conversation_id}`
        
        **📊 Token Usage**
        - Total: **{st.session_state.total_tokens:,}**
        """)
        
        # Blueprint summary
        blueprint = st.session_state.conversation_blueprint
        with st.expander("🔍 Blueprint"):
            st.write(f"Topic: {blueprint['main_topic']}")
            st.write(f"Questions: {blueprint['total_questions']}")
        
        # Memory upload
        st.markdown("---")
        st.markdown("### 📂 Load Memory")
        uploaded_memory = st.file_uploader("Upload .md memory", type=['md'])
        
        if uploaded_memory:
            with st.spinner("Loading..."):
                memory_data = load_memory_file(uploaded_memory)
                if "error" not in memory_data:
                    st.success(f"Loaded: {memory_data.get('username', 'Unknown')}")
                    if memory_data not in st.session_state.loaded_memories:
                        st.session_state.loaded_memories.append(memory_data)
        
        # Download memory
        st.markdown("---")
        if st.button("💾 Download Memory", use_container_width=True):
            if st.session_state.messages:
                filepath, filename = save_memory_to_file()
                with open(filepath, 'r') as f:
                    st.download_button(
                        label="📥 Save Memory",
                        data=f.read(),
                        file_name=filename,
                        mime="text/markdown"
                    )
            else:
                st.warning("No conversation yet!")

# ============================================
# SETTINGS: AI PROVIDER CONFIG
# ============================================
def render_ai_settings():
    """Render AI provider settings"""
    config = get_config()
    
    st.markdown("### ⚙️ AI Provider Settings")
    
    # Provider selection
    providers = ["openai", "ollama", "anthropic", "custom"]
    provider_labels = {
        "openai": "🔷 OpenAI (GPT)",
        "ollama": "🟠 Ollama (Local)",
        "anthropic": "🟣 Anthropic (Claude)",
        "custom": "⚪ Custom API"
    }
    
    selected_provider = st.selectbox(
        "AI Provider",
        providers,
        index=providers.index(config.get("provider", "openai")),
        format_func=lambda x: provider_labels.get(x, x)
    )
    config["provider"] = selected_provider
    
    # Provider-specific settings
    if selected_provider == "openai":
        with st.expander("🔷 OpenAI Settings", expanded=True):
            config["openai"]["api_key"] = st.text_input(
                "API Key",
                type="password",
                value=config["openai"].get("api_key", ""),
                help="Get from https://platform.openai.com/api-keys"
            )
            config["openai"]["model"] = st.selectbox(
                "Model",
                ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"].index(
                    config["openai"].get("model", "gpt-3.5-turbo")
                )
            )
            config["openai"]["base_url"] = st.text_input(
                "Base URL",
                value=config["openai"].get("base_url", "https://api.openai.com/v1/chat/completions")
            )
    
    elif selected_provider == "ollama":
        with st.expander("🟠 Ollama Settings", expanded=True):
            config["ollama"]["base_url"] = st.text_input(
                "Base URL",
                value=config["ollama"].get("base_url", "http://localhost:11434"),
                help="Default: http://localhost:11434"
            )
            config["ollama"]["model"] = st.text_input(
                "Model",
                value=config["ollama"].get("model", "llama3.2"),
                help="Examples: llama3.2, mistral, phi3, gemma2"
            )
            
            # Check connection button
            if st.button("🔗 Test Ollama Connection"):
                try:
                    response = requests.get(
                        f"{config['ollama']['base_url']}/api/tags",
                        timeout=5
                    )
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        st.success(f"✅ Connected! Available models: {len(models)}")
                        for m in models[:5]:
                            st.write(f"  - {m.get('name', 'Unknown')}")
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
    
    elif selected_provider == "anthropic":
        with st.expander("🟣 Anthropic Settings", expanded=True):
            config["anthropic"]["api_key"] = st.text_input(
                "API Key",
                type="password",
                value=config["anthropic"].get("api_key", ""),
                help="Get from https://console.anthropic.com/"
            )
            config["anthropic"]["model"] = st.selectbox(
                "Model",
                ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241007", "claude-3-opus-20240229"],
                index=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241007", "claude-3-opus-20240229"].index(
                    config["anthropic"].get("model", "claude-3-5-haiku-20241007")
                )
            )
    
    elif selected_provider == "custom":
        with st.expander("⚪ Custom API Settings", expanded=True):
            config["custom"]["base_url"] = st.text_input(
                "Base URL",
                value=config["custom"].get("base_url", ""),
                placeholder="https://api.example.com/v1/chat"
            )
            config["custom"]["api_key"] = st.text_input(
                "API Key",
                type="password",
                value=config["custom"].get("api_key", "")
            )
            config["custom"]["model"] = st.text_input(
                "Model Name",
                value=config["custom"].get("model", ""),
                placeholder="model-name"
            )
            config["custom"]["payload_template"] = st.text_area(
                "Payload Template",
                value=config["custom"].get("payload_template", '{"model":"{model}","messages":[{"role":"user","content":"{prompt}"}]}'),
                help="Use {prompt} and {model} as placeholders"
            )
    
    # Embedding settings
    st.markdown("---")
    st.markdown("### 🔧 Embedding Settings")
    embedding_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"],
        help="all-MiniLM-L6-v2 is fast and accurate"
    )
    
    # Save button
    if st.button("💾 Save Settings"):
        save_config(config)
        st.success("Settings saved!")
        
        # Rebuild QA chain with new settings
        if st.session_state.chunks:
            with st.spinner("Rebuilding QA chain..."):
                process_documents(
                    [],  # No new files, just rebuild
                    embeddings_model=embedding_model
                )
                st.success("QA chain updated!")
    
    return config, embedding_model

# ============================================
# CHAT INTERFACE
# ============================================
def render_chat():
    """Render main chat interface"""
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- {src}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.qa_chain is None:
                        response = "Please upload documents first!"
                    else:
                        result = st.session_state.qa_chain({"query": prompt})
                        response = result["result"]
                        
                        sources = []
                        if "source_documents" in result:
                            for doc in result["source_documents"]:
                                src = doc.metadata.get("source", "Unknown")
                                if src not in sources:
                                    sources.append(src)
                        
                        st.session_state.input_tokens += len(prompt.split()) * 2
                        st.session_state.output_tokens += len(response.split()) * 2
                        st.session_state.total_tokens = (st.session_state.input_tokens + 
                                                          st.session_state.output_tokens)
                        
                        update_blueprint(prompt)
                        
                        message_data = {"role": "assistant", "content": response}
                        if sources:
                            message_data["sources"] = sources
                        
                        st.session_state.messages.append(message_data)
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for src in sources:
                                    st.markdown(f"- {src}")
                
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(response)
        
        st.rerun()

# ============================================
# MAIN APP
# ============================================
def main():
    st.set_page_config(
        page_title="RAG Q&A System",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("🤖 RAG Q&A System with AI Memory")
    st.markdown("*Supports OpenAI, Ollama, Anthropic & Custom APIs*")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Documents", "📊 Stats", "⚙️ Settings"])
    
    with tab1:
        # Document upload
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['csv', 'txt', 'md'],
                accept_multiple_files=True
            )
        with col2:
            st.markdown("### 📊 Status")
            stats = st.session_state.document_stats
            st.write(f"Documents: {stats['total_docs']}")
            st.write(f"Chunks: {stats['total_chunks']}")
        
        if uploaded_files and st.button("🚀 Process", type="primary"):
            with st.spinner("Processing..."):
                config = get_config()
                if process_documents(uploaded_files):
                    st.success(f"✅ Processed {len(uploaded_files)} files!")
                    st.rerun()
        
        st.markdown("---")
        render_chat()
    
    with tab2:
        st.markdown("### 📄 Document Management")
        if st.session_state.chunks:
            st.success(f"📚 {len(st.session_state.chunks)} chunks loaded")
            
            with st.expander("🔍 View Chunks"):
                for i, chunk in enumerate(st.session_state.chunks[:10]):
                    st.markdown(f"**Chunk {i+1}:** {chunk.page_content[:200]}...")
        else:
            st.info("No documents. Go to Chat tab to upload.")
    
    with tab3:
        st.markdown("### 📈 Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Tokens", st.session_state.total_tokens)
        with col3:
            st.metric("Questions", st.session_state.conversation_blueprint["total_questions"])
        
        # Token chart
        if st.session_state.input_tokens > 0:
            fig = px.pie(
                values=[st.session_state.input_tokens, st.session_state.output_tokens],
                names=['Input', 'Output'],
                title='Token Distribution'
            )
            st.plotly_chart(fig)
    
    with tab4:
        config, embedding_model = render_ai_settings()
        
        st.markdown("---")
        st.markdown("### 🧹 Session Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Memory Files"):
                import shutil
                if os.path.exists(MEMORY_DIR):
                    shutil.rmtree(MEMORY_DIR)
                    os.makedirs(MEMORY_DIR)
                st.success("Memory cleared!")
    
    # Sidebar
    render_sidebar_memory()
    
    st.markdown("---")
    st.caption("Creator Mohammad Saeed Angiz")

if __name__ == "__main__":
    main()
