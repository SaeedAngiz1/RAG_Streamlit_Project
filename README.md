RAG Streamlit Project
# RAG Q&A System with AI Memory

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-FF6F61?style=for-the-badge&logo=LangChain&logoColor=white)

> A powerful RAG (Retrieval-Augmented Generation) Q&A system built with Streamlit, supporting multiple AI providers with intelligent conversation memory.

## 🚀 Features

### Core Features
- 📄 **Multi-format Document Support** - Upload CSV, TXT, MD files
- 🔍 **RAG Pipeline** - LangChain + ChromaDB for intelligent retrieval
- 🤖 **Multiple AI Providers** - OpenAI, Ollama, Anthropic, Custom APIs
- 💾 **AI Memory System** - Auto-generate downloadable session memories
- 📊 **Token Tracking** - Real-time token usage monitoring
- 🔍 **Conversation Blueprint** - AI-powered conversation flow analysis

### AI Provider Support

| Provider | API Key | Local | Model Examples |
|----------|---------|-------|---------------|
| OpenAI | ✅ Required | ❌ | GPT-4o, GPT-4-turbo, GPT-3.5-turbo |
| Ollama | ❌ | ✅ | Llama3.2, Mistral, Phi3, Gemma2 |
| Anthropic | ✅ Required | ❌ | Claude-3.5-Sonnet, Claude-3-Haiku |
| Custom | ⚠️ Optional | ❌ | Any OpenAI-compatible API |

### Memory System
- 📝 **Auto-generated** session summaries
- 💾 **Downloadable** .md memory files
- 📂 **Uploadable** previous memory files
- 🔍 **Blueprint tracking** - topic flow, entities, question types

## 📦 Installation

### Prerequisites
- Python 3.9+
- pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/SaeedAngiz1/RAG_Streamlit_Project.git
cd RAG_Streamlit_Project

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
🔧 Configuration
OpenAI Setup
Get your API key from OpenAI Platform
Go to Settings tab
Select OpenAI as provider
Enter your API key
Choose your model (GPT-4o recommended)
Ollama Setup (Free Local Models)
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Download models
ollama pull llama3.2
ollama pull mistral
ollama pull phi3
ollama pull gemma2

# Start Ollama server
ollama serve
Then in the app:

Go to Settings → Select Ollama
Base URL: http://localhost:11434
Model: llama3.2 (or your preferred model)
Click Test Connection to verify
Anthropic Setup
Get your API key from Anthropic Console
Go to Settings → Select Anthropic
Enter your API key
Choose Claude model
📁 Project Structure
rag_streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── memory_files/          # Saved memory files
│   └── *.md              # Session memory documents
└── chroma_db/            # Vector database (auto-created)
📖 Usage Guide
1. Document Upload
Navigate to the Chat tab
Upload your documents (CSV, TXT, or MD)
Click Process to create the knowledge base
Wait for embedding creation to complete
2. Ask Questions
Type your question in the chat input
AI retrieves relevant chunks from your documents
Generate contextual answers
View sources by expanding the Sources section
3. Memory System
Auto-Save Memory
Memory auto-generates as you chat
Click Download Memory in sidebar to save .md file
Includes: username, timestamps, tokens, conversation flow
Load Previous Memory
In sidebar, use Upload Memory file picker
Select a previously saved .md memory file
AI loads and displays memory details
4. Conversation Blueprint
The AI automatically tracks:

Main Topic - Primary subject of conversation
Key Entities - Important terms and concepts
Question Types - Definition, procedure, explanation, comparison
Conversation Flow - Timeline of exchanges
📊 Token Tracking
Metric	Description
Input Tokens	Tokens used for user queries + context
Output Tokens	Tokens in AI responses
Total Tokens	Combined input + output
Estimated Cost	Cost based on GPT-3.5 pricing
🎨 Interface Overview
Main Tabs
Tab	Purpose
💬 Chat	Main Q&A interface with document upload
📄 Documents	View and manage uploaded documents
📊 Stats	Token usage and conversation statistics
⚙️ Settings	AI provider and embedding configuration
Sidebar Features
👤 Username input
📅 Session information
📊 Token counter
🔍 Blueprint summary
📂 Memory upload/download
💾 Save session memory
🔧 Memory File Format
# AI Session Memory

## User Information
- **Username:** John Doe
- **Session ID:** a1b2c3d4
- **Session Start:** 2024-01-15 10:30:00
- **Duration:** 00:45:23

## Token Usage
- **Input Tokens:** 1,250
- **Output Tokens:** 890
- **Total Tokens:** 2,140
- **Estimated Cost:** $0.004

## Conversation Blueprint
- **Main Topic:** Machine Learning
- **Total Questions:** 5
- **Question Types:** definition, procedure

### Key Entities
- neural networks
- transformers
- embeddings

### Conversation Flow
| # | Timestamp | Question Preview |
|---|-----------|------------------|
| 1 | 10:30:15 | What is deep learning? |
| 2 | 10:32:45 | How do transformers work? |

## Chat History
### User
What is deep learning?

### AI
Deep learning is a specialized form of machine learning...
🛠️ API Configuration
Custom API Template
For custom API providers, use this payload template:

{
  "model": "{model}",
  "messages": [
    {
      "role": "user",
      "content": "{prompt}"
    }
  ]
}
Available placeholders:

{model} - Your model name
{prompt} - User's question
🔐 Security Notes
API keys are stored in session state only
Keys are never saved to disk
Memory files contain no sensitive data by default
Use environment variables for production deployments
🐛 Troubleshooting
Common Issues
Issue	Solution
"No module named 'streamlit'"	Run pip install -r requirements.txt
Ollama connection failed	Ensure Ollama is running (ollama serve)
Document processing slow	Try a smaller embedding model
Token limit exceeded	Split large documents into smaller files
Ollama Issues
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
📝 Requirements
streamlit>=1.35.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-openai>=0.0.20
sentence-transformers>=2.2.0
chromadb>=0.4.0
openai>=1.0.0
tiktoken>=0.5.0
plotly>=5.20.0
pandas>=2.0.0
requests>=2.31.0
🚀 Deployment
Streamlit Cloud
⚠️ Note: TensorFlow and some ML libraries don't work on Streamlit Cloud's free tier.

Push to GitHub
Connect to Streamlit Cloud
Deploy with minimal requirements
Render.com
# Build Command
pip install -r requirements.txt

# Start Command
streamlit run app.py
Local Deployment
# Set API key (optional)
export OPENAI_API_KEY="your-key-here"

# Run
streamlit run app.py --server.port 8501
📚 Technologies Used
Technology	Purpose
Streamlit	Web interface
LangChain	LLM orchestration
ChromaDB	Vector database
HuggingFace	Embeddings
SentenceTransformers	Text embeddings
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing)
Open a Pull Request
📄 License
This project is open source and available under the MIT License [blocked].

👤 Author
Mohammad Saeed Angiz

GitHub: @SaeedAngiz1
Project: RAG Streamlit
🙏 Acknowledgments
LangChain Community for excellent documentation
Streamlit for the amazing web framework
HuggingFace for free embedding models
⭐ If this project helped you, please give it a star!