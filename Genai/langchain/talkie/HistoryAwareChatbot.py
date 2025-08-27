import json
from datetime import datetime

from typing import Dict, List, Optional
import logging


import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from helper import clean_json_string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoryAwareChatbot:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", max_tokens: int = 2000):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens

        # Configure Google GenAI client
        genai.configure(api_key=api_key)
        self.client = genai

        # LLM with optimized settings
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )

        # LangChain memory
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        # Conversation + facts
        self.conversation_history = []
        self.facts = {}

        # Semantic memory (FAISS) - Initialize properly
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Specify device
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        # Initialize with empty vector store properly
        try:
            self.vector_db = FAISS.from_texts(["initialization"], self.embeddings)
            self.vector_db.delete([0])  # Remove initialization text
        except:
            # Fallback if delete doesn't work
            self.vector_db = None

        # Base instruction - more specific
        self.base_instruction = (
            "You are a helpful assistant with perfect memory. Use the background information "
            "naturally in your responses without explicitly mentioning it. Be conversational "
            "and build upon previous interactions."
        )

    def _initialize_vector_db(self):
        """Lazy initialization of vector DB when first needed."""
        if self.vector_db is None:
            dummy_text = ["Initial conversation context"]
            self.vector_db = FAISS.from_texts(dummy_text, self.embeddings)

    # -----------------------------
    # Core chat
    # -----------------------------
    def chat(self, user_input: str) -> Dict:
        timestamp = datetime.now().isoformat()
        self.conversation_history.append({
            "role": "user", 
            "content": user_input, 
            "timestamp": timestamp
        })

        # Check and compress if needed
        current_tokens = self.count_tokens()
        if current_tokens > self.max_tokens:
            logger.info(f"Token limit exceeded ({current_tokens}). Compressing history...")
            self.compress_history()

        try:
            # Memory context
            memory_context = self.retrieve_memory(user_input)

            # Construct messages more efficiently
            full_context = f"{self.base_instruction}\n\nBackground context: {memory_context}\n\nUser message: {user_input}"
            
            response = self.llm.invoke(full_context).content
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": response, 
                "timestamp": datetime.now().isoformat()
            })

            # Update memory after successful response
            self.update_memory()

            return {
                "response": response,
                "conversation_length": len(self.conversation_history),
                "timestamp": timestamp,
                "tokens_used": self.count_tokens(),
                "facts_count": len(self.facts)
            }

        except Exception as e:
            error_response = f"I apologize, but I encountered an error. Please try again."
            logger.error(f"Chat error: {str(e)}")
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": error_response, 
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "response": error_response,
                "error": str(e),
                "conversation_length": len(self.conversation_history),
                "timestamp": timestamp,
            }

    # -----------------------------
    # Token counting - optimized
    # -----------------------------
    def count_tokens(self) -> int:
        """Count tokens more efficiently."""
        try:
            # Only count recent messages for efficiency
            recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
            
            contents = []
            for h in recent_history:
                contents.append({
                    "role": h["role"],
                    "parts": [h["content"]]
                })
            
            model = self.client.GenerativeModel(self.model_name)
            count = model.count_tokens(contents)
            return count.total_tokens
            
        except Exception as e:
            logger.warning(f"Token count failed: {e}, using estimation")
            # Fallback estimation: ~4 chars per token
            total_chars = sum(len(h["content"]) for h in self.conversation_history)
            return total_chars // 4

    # -----------------------------
    # Memory management - optimized
    # -----------------------------
    def update_memory(self):
        """Extract facts and update semantic memory."""
        # Only process if we have new conversation
        if len(self.conversation_history) < 2:
            return

        try:
            # More focused fact extraction
            recent_conversation = self.conversation_history[-4:]  # Last 2 exchanges
            fact_prompt = f"""
            Extract ONLY new facts about the user from this conversation snippet.
            Focus on: preferences, personal details, goals, interests.
            Return as JSON object or empty object if no new facts.
            
            Conversation: {json.dumps(recent_conversation, indent=2)}
            
            Example: {{"name": "John", "likes": "pizza", "job": "teacher"}}
            """
            
            result = self.llm.invoke(fact_prompt).content.strip()
            result_clean = clean_json_string(result)

            try:
                new_facts = json.loads(result_clean)
                if isinstance(new_facts, dict):
                    # Only update if we have meaningful facts
                    meaningful_facts = {k: v for k, v in new_facts.items() 
                                      if v and str(v).strip() and len(str(v).strip()) > 2}
                    self.facts.update(meaningful_facts)
                    
                    if meaningful_facts:
                        logger.info(f"Updated facts: {meaningful_facts}")
                        
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from fact extraction: {result_clean}")

        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")

        # Update semantic memory
        self._update_semantic_memory()

    def _update_semantic_memory(self):
        """Update vector database with conversation snippets."""
        if len(self.conversation_history) >= 2:
            try:
                self._initialize_vector_db()
                
                # Create meaningful snippet
                user_msg = self.conversation_history[-2]['content']
                bot_msg = self.conversation_history[-1]['content']
                
                # Add context and timestamp
                snippet = f"[{datetime.now().strftime('%Y-%m-%d')}] User: {user_msg} | Response: {bot_msg}"
                
                self.vector_db.add_texts([snippet])
                logger.debug("Added snippet to semantic memory")
                
            except Exception as e:
                logger.error(f"Semantic memory update failed: {e}")

    def retrieve_memory(self, query: str) -> str:
        """Retrieve relevant context from both structured facts and semantic memory."""
        context_parts = []
        
        # Add structured facts
        if self.facts:
            fact_str = json.dumps(self.facts, indent=2)
            context_parts.append(f"User facts: {fact_str}")
        
        # Add semantic memories
        if self.vector_db is not None:
            try:
                docs = self.vector_db.similarity_search(query, k=3)
                if docs:
                    semantic_memories = [d.page_content for d in docs if d.page_content.strip()]
                    if semantic_memories:
                        context_parts.append(f"Relevant past conversations: {'; '.join(semantic_memories)}")
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        return " | ".join(context_parts) if context_parts else "No prior context available."

    def compress_history(self):
        """Intelligent history compression preserving important information."""
        if len(self.conversation_history) <= 4:
            return  # Don't compress if history is short
            
        try:
            # Keep first 2 and last 2 messages, summarize the middle
            start_messages = self.conversation_history[:2]
            end_messages = self.conversation_history[-2:]
            middle_messages = self.conversation_history[2:-2]
            
            if middle_messages:
                # Summarize middle section
                summary_prompt = (
                    "Summarize this conversation section concisely, focusing on key topics "
                    "and user preferences mentioned: " + 
                    json.dumps(middle_messages, indent=2)
                )
                
                summary = self.llm.invoke(summary_prompt).content
                
                # Create compressed history
                summary_message = {
                    "role": "system", 
                    "content": f"[Conversation summary]: {summary}", 
                    "timestamp": datetime.now().isoformat()
                }
                
                self.conversation_history = start_messages + [summary_message] + end_messages
                
            # Clear LangChain memory and add summary
            self.memory.clear()
            fact_context = json.dumps(self.facts, indent=2) if self.facts else "No facts recorded yet."
            self.memory.chat_memory.add_ai_message(f"Context preserved. Known facts: {fact_context}")
            
            logger.info("History compressed successfully")
            
        except Exception as e:
            logger.error(f"History compression failed: {e}")
            # Fallback: just keep recent messages
            self.conversation_history = self.conversation_history[-6:]

    # -----------------------------
    # Utilities
    # -----------------------------
    def get_conversation_summary(self) -> str:
        """Get a comprehensive conversation summary."""
        if not self.conversation_history:
            return "No conversation history available."
            
        try:
            summary_prompt = (
                "Provide a comprehensive summary of this conversation, including main topics, "
                "user preferences discovered, and key points discussed: " + 
                json.dumps(self.conversation_history, indent=2)
            )
            return self.llm.invoke(summary_prompt).content
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Unable to generate summary: {str(e)}"

    def clear_conversation_history(self):
        """Clear all conversation data."""
        self.conversation_history = []
        self.memory.clear()
        self.facts = {}
        
        # Reinitialize vector DB
        try:
            self.vector_db = FAISS.from_texts(["initialization"], self.embeddings)
            self.vector_db.delete([0])
        except:
            self.vector_db = None
            
        logger.info("All memory cleared")

    def get_memory_stats(self) -> Dict:
        """Get current memory usage statistics."""
        return {
            "conversation_length": len(self.conversation_history),
            "facts_count": len(self.facts),
            "estimated_tokens": self.count_tokens(),
            "vector_db_initialized": self.vector_db is not None
        }

