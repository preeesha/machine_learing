# 🤖 Talkie - Advanced Memory-Aware Chatbot

A sophisticated chatbot system that combines **semantic memory** (FAISS vector search) with **regular memory** (structured facts) to provide contextually aware conversations. Built with LangChain and Google's Gemini AI.

## 🚀 Features

- **Dual Memory System**: Combines semantic search with structured fact extraction
- **Intelligent History Compression**: Automatically manages conversation length while preserving context
- **Real-time Memory Updates**: Continuously learns and adapts from conversations
- **Token Management**: Efficient token counting and automatic history compression
- **Persistent Memory**: FAISS vector database for semantic similarity search
- **Structured Fact Extraction**: Automatically identifies and stores user preferences and details

## 🏗️ Architecture

### Memory Systems

#### 1. **Semantic Memory (FAISS Vector Database)**
- **Purpose**: Stores conversation snippets for semantic similarity search
- **Technology**: FAISS with HuggingFace embeddings (`all-MiniLM-L6-v2`)
- **Efficiency**: 
  - Lazy initialization to reduce startup time
  - Normalized embeddings for better similarity matching
  - CPU-optimized for broader compatibility
  - Retrieves top 3 most relevant past conversations

#### 2. **Regular Memory (Structured Facts)**
- **Purpose**: Stores extracted user preferences, personal details, and facts
- **Technology**: JSON-based structured storage with LLM-powered extraction
- **Efficiency**:
  - Focused fact extraction from recent conversations only
  - Meaningful fact filtering (length > 2 characters)
  - Automatic JSON cleaning and validation

### Memory Management Efficiency

- **Smart Compression**: Keeps first 2 and last 2 messages, summarizes middle sections
- **Token Optimization**: Only counts recent messages (last 10) for efficiency
- **Context Preservation**: Maintains critical information during compression
- **Memory Stats**: Real-time monitoring of conversation length, facts count, and token usage

## 📋 Prerequisites

- Python 3.8+
- Google AI API key
- Required packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   cd Genai/langchain/talkie
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

## 🚀 Usage

### Command Line Interface

Run the main chatbot:
```bash
python v1.py
```

### Available Commands

| Command | Description |
|---------|-------------|
| `quit` | Exit the chatbot |
| `summary` | Get conversation summary |
| `clear` | Clear all memory and history |
| `tokens` | Show current token count |
| `facts` | Display extracted facts |
| `stats` | Show memory statistics |

### Example Session

```bash
🤖 Enhanced Gemini Hybrid-Memory Chatbot ready!
Commands: 'quit', 'summary', 'clear', 'tokens', 'facts', 'stats'

👤 You: hello, my name is John and I love pizza
🤖 Assistant: Hello John! It's great to meet you. I can see you're a pizza enthusiast - that's wonderful!

👤 You: facts
📌 Known facts: {
  "name": "John",
  "likes": "pizza"
}

👤 You: what should we talk about?
🤖 Assistant: Since you mentioned loving pizza, we could discuss your favorite pizza places, toppings, or maybe explore other Italian cuisine! What interests you most?
```
### Memory Settings

- **Conversation History**: Automatically compressed when exceeding token limit
- **Fact Extraction**: Processes last 4 messages for new information
- **Semantic Search**: Retrieves top 3 most relevant past conversations
- **Vector Database**: FAISS with sentence-transformers embeddings

## 📊 Performance Features

### Token Management
- **Efficient Counting**: Only counts recent messages (last 10)
- **Automatic Compression**: Triggers when token limit exceeded
- **Fallback Estimation**: Character-based estimation if token counting fails

### Memory Optimization
- **Lazy Initialization**: FAISS database initialized only when needed
- **Focused Processing**: Only processes recent conversations for fact extraction
- **Smart Summarization**: Preserves context while reducing memory footprint

### Error Handling
- **Graceful Degradation**: Continues operation even if memory operations fail
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Fallback Mechanisms**: Multiple fallback strategies for critical operations

## 🧠 How Memory Systems Work Together

### 1. **Input Processing**
```
User Input → Token Count Check → Memory Retrieval → Context Construction
```

### 2. **Memory Retrieval**
```
Query → Semantic Search (FAISS) + Structured Facts → Combined Context
```

### 3. **Response Generation**
```
Context + User Input → LLM Processing → Response + Memory Update
```

### 4. **Memory Update**
```
New Conversation → Fact Extraction → Semantic Memory Update → History Compression (if needed)
```

## 🔍 Memory Efficiency Explained

### **Semantic Memory Efficiency**
- **Vector Similarity**: FAISS provides sub-linear search complexity
- **Embedding Optimization**: Normalized embeddings improve similarity accuracy
- **Lazy Loading**: Database only initialized when first needed
- **Batch Processing**: Multiple texts added simultaneously

### **Regular Memory Efficiency**
- **Selective Extraction**: Only processes recent conversations (last 4 messages)
- **Meaningful Filtering**: Discards empty or very short facts
- **JSON Validation**: Ensures data integrity before storage
- **Automatic Cleanup**: Removes invalid or corrupted data

### **Combined Benefits**
- **Contextual Relevance**: Semantic search finds conversationally relevant information
- **Structured Access**: Facts provide quick access to user preferences
- **Memory Compression**: Intelligent summarization preserves important context
- **Scalability**: Both systems scale independently based on usage patterns

## 📁 Project Structure

```
talkie/
├── HistoryAwareChatbot.py    # Main chatbot class with dual memory
├── v1.py                     # Command-line interface
├── helper.py                 # Utility functions
├── chatbot.ipynb            # Jupyter notebook examples
├── faiss_index/             # Vector database storage
├── res/                     # Chat history files
└── README.md                # This file
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```bash
   ❌ Please set GOOGLE_API_KEY in your .env file
   ```
   **Solution**: Create `.env` file with your Google AI API key

2. **Memory Initialization Failed**
   ```bash
   Semantic memory update failed: [Error details]
   ```
   **Solution**: Check if FAISS index directory is writable

3. **High Token Usage Warning**
   ```bash
   ⚠️  High token usage: [token_count]
   ```
   **Solution**: This is informational - the system automatically compresses history

### Performance Tips

- **Monitor Memory Stats**: Use `stats` command to track memory usage
- **Regular Cleanup**: Use `clear` command to reset memory when needed
- **Fact Review**: Use `facts` command to verify extracted information
- **Summary Generation**: Use `summary` command for conversation overview

## 🔮 Future Enhancements

- **Persistent Storage**: Save memory to disk for long-term persistence
- **Memory Analytics**: Advanced insights into memory usage patterns
- **Custom Embeddings**: Support for different embedding models
- **Memory Export/Import**: Backup and restore conversation memory
- **Multi-User Support**: Separate memory spaces for different users

## 📚 Technical Details

### Dependencies
- **LangChain**: Core framework for LLM interactions
- **FAISS**: Vector similarity search
- **HuggingFace**: Sentence embeddings
- **Google GenAI**: LLM provider
- **Python-dotenv**: Environment variable management

### Memory Algorithms
- **Similarity Search**: Cosine similarity with normalized embeddings
- **Fact Extraction**: LLM-powered structured information extraction
- **History Compression**: Intelligent summarization preserving context
- **Token Management**: Efficient counting and automatic compression

## 🤝 Contributing

This project follows clean code principles and design patterns. When contributing:

1. Maintain the dual memory architecture
2. Follow existing error handling patterns
3. Add comprehensive logging for new features
4. Ensure backward compatibility
5. Update this README for new features

## 📄 License

This project is part of the learning repository and follows the same licensing terms.

---

**Built with ❤️ using LangChain, FAISS, and Google's Gemini AI**
