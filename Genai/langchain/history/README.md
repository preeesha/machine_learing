# History-Aware Chatbot with LangChain and Google Gemini

A sophisticated chatbot implementation that combines LangChain's conversational AI capabilities with Google Gemini models and history-aware document retrieval.

## Features

- 🤖 **Google Gemini Integration**: Powered by Google's latest AI models
- 📚 **Document Retrieval**: Context-aware responses using vector embeddings
- 💾 **Conversation Memory**: Maintains conversation context and history
- 📊 **History Tracking**: Complete conversation logging with timestamps
- 🔍 **Smart Retrieval**: FAISS vector store for efficient document search
- 💾 **Persistence**: Save and load conversation histories
- 📈 **Analytics**: Conversation statistics and summaries

## Architecture

```
User Input → Memory Buffer → Document Retriever → Gemini LLM → Response
                ↓              ↓
        Conversation History  Vector Store
                ↓              ↓
        Timestamp Logging   Embeddings
```

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd Genai/langchain/history
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
    ```bash
    export GOOGLE_API_KEY="your_google_api_key_here"
    ```

## Configuration

The chatbot can be configured through environment variables or by modifying `config.py`:

- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `MODEL_NAME`: Gemini model to use (default: gemini-1.5-flash)
- `TEMPERATURE`: Response creativity (0.0-1.0)
- `MEMORY_WINDOW_SIZE`: Number of exchanges to remember
- `CHUNK_SIZE`: Document chunk size for vectorization
- `RETRIEVER_K`: Number of relevant documents to retrieve

## Usage

### Basic Usage

```python
from advanced_chatbot import HistoryAwareChatbot

# Initialize chatbot
chatbot = HistoryAwareChatbot(api_key="your_api_key")

# Start chatting
response = chatbot.chat("Hello, how are you?")
print(response["response"])
```

### With Document Retrieval

```python
# Add documents for context-aware responses
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language."
]

chatbot.add_documents(documents)

# Chat with document context
response = chatbot.chat("What is deep learning?")
print(response["response"])
```

### Interactive Mode

Run the chatbot in interactive mode:

```bash
python advanced_chatbot.py
```

**Commands:**
- `summary`: Get conversation summary
- `stats`: View conversation statistics
- `save`: Save conversation history
- `clear`: Clear conversation history
- `quit`: Exit chatbot

## API Reference

### HistoryAwareChatbot Class

#### Methods

- `__init__(api_key, model_name)`: Initialize chatbot
- `add_documents(documents, chunk_size, chunk_overlap)`: Add documents for retrieval
- `chat(user_input)`: Process user input and generate response
- `get_conversation_summary()`: Generate conversation summary
- `save_conversation_history(filename)`: Save history to file
- `load_conversation_history(filename)`: Load history from file
- `clear_conversation_history()`: Clear all history
- `get_conversation_stats()`: Get conversation statistics

#### Properties

- `conversation_history`: List of all conversation exchanges
- `memory`: LangChain memory object
- `vector_store`: FAISS vector store for document retrieval

## Example Output

```
🤖 History-Aware Chatbot initialized!
Type 'quit' to exit, 'summary' for conversation summary, 'stats' for statistics
Type 'save' to save conversation, 'clear' to clear history
--------------------------------------------------

👤 You: What is machine learning?

🤖 Assistant: Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. It enables computers to learn and improve from experience without being explicitly programmed for every task.

📚 Sources:
  1. Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models...
```

## File Structure

```
history/
├── advanced_chatbot.py      # Main chatbot implementation
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── histroy_tracking.ipynb  # Original notebook
```

## Dependencies

- **LangChain**: Conversational AI framework
- **Google Generative AI**: Gemini model integration
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Open-source embeddings for document vectorization
- **Python Standard Library**: JSON, datetime, typing

## Error Handling

The chatbot includes comprehensive error handling:

- API key validation
- Network error recovery
- Document processing errors
- Memory management issues
- Graceful fallbacks for missing components

## Best Practices

1. **API Key Security**: Never commit API keys to version control
2. **Memory Management**: Use appropriate window sizes for your use case
3. **Document Chunking**: Optimize chunk sizes for your document types
4. **Error Monitoring**: Check conversation logs for issues
5. **Resource Cleanup**: Clear history when no longer needed

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `GOOGLE_API_KEY` is set correctly for Gemini
2. **Import Errors**: Verify all dependencies are installed
3. **Memory Issues**: Reduce `MEMORY_WINDOW_SIZE` for long conversations
4. **Vector Store Errors**: Check document format and chunk sizes

### Performance Tips

- Use smaller chunk sizes for faster processing
- Limit memory window size for long conversations
- Optimize document retrieval with appropriate `k` values
- Monitor conversation length for memory usage

## License

This project is for educational and research purposes. Please ensure compliance with Google's API terms of service.

## Contributing

Feel free to enhance the chatbot with additional features:

- Custom memory implementations
- Additional vector store backends
- Enhanced conversation analytics
- Multi-modal capabilities
- Streaming responses
