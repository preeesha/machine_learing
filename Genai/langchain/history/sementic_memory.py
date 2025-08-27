import json
from datetime import datetime
import re
import os 
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Helper: clean LLM outputs
# -----------------------------
def clean_json_string(s: str) -> str:
    s = re.sub(r"^```(?:json)?\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()


# -----------------------------
# Main chatbot class
# -----------------------------
class HistoryAwareChatbot:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

        # Configure Google GenAI client
        genai.configure(api_key=api_key)
        self.client = genai

        # LLM with fix for system messages
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )

        # LangChain memory
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.chain = ConversationChain(llm=self.llm, memory=self.memory)

        # Conversation + facts
        self.conversation_history = []
        self.facts = {}

        # Semantic memory (FAISS) - start empty (no dummy!)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = FAISS.from_texts([""], self.embeddings)

        # Base instruction
        self.base_instruction = (
            "You are a helpful assistant. You remember the user across conversations. "
            "You have access to background facts and past chat summaries, but you should "
            "never repeat them back directly. Use them naturally to make your replies smarter."
        )


    # -----------------------------
    # Core chat
    # -----------------------------
    def chat(self, user_input: str) -> dict:
        timestamp = datetime.now().isoformat()
        self.conversation_history.append({"role": "user", "content": user_input, "timestamp": timestamp})

        # Compress if tokens too high
        if self.count_tokens() > 200:
            print("⚠️ Token limit exceeded. Compressing history...")
            self.compress_history()
            self.conversation_history.append({"role": "user", "content": user_input, "timestamp": timestamp})

        try:
            # Memory context
            memory_context = self.retrieve_memory(user_input)

            # Messages construction (background info hidden)
            messages = [
                {"role": "user", "content": self.base_instruction},
                {"role": "user", "content": f"(Background info for you: {memory_context})"},
                {"role": "user", "content": user_input}
            ]

            response = self.llm.invoke(messages).content
            self.conversation_history.append(
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            )

            # Update memory
            self.update_memory()

            return {
                "response": response,
                "conversation_length": len(self.conversation_history),
                "timestamp": timestamp,
                "tokens_used": self.count_tokens(),
            }

        except Exception as e:
            error_response = f"Error: {str(e)}"
            self.conversation_history.append(
                {"role": "assistant", "content": error_response, "timestamp": datetime.now().isoformat()}
            )
            return {
                "response": error_response,
                "error": str(e),
                "conversation_length": len(self.conversation_history),
                "timestamp": timestamp,
            }


    # -----------------------------
    # Token counting
    # -----------------------------
    def count_tokens(self) -> int:
        try:
            contents = [{"role": h["role"], "parts": [h["content"]]} for h in self.conversation_history]
            model = self.client.GenerativeModel(self.model_name)
            count = model.count_tokens(contents)
            return count.total_tokens
        except Exception as e:
            print(f"⚠️ Token count failed: {e}")
            return -1


    # -----------------------------
    # Memory management
    # -----------------------------
    def update_memory(self):
        """Extract structured facts + add semantic snippet."""
        try:
            fact_prompt = f"""
            From this conversation, extract key facts about the user 
            (name, favorites, dislikes, personality, preferences).
            Respond ONLY in valid JSON.
            Conversation: {json.dumps(self.conversation_history, indent=2)}
            """
            result = self.llm.invoke(fact_prompt).content.strip()
            result_clean = clean_json_string(result)

            try:
                new_facts = json.loads(result_clean)
                new_facts = {k: v for k, v in new_facts.items() if v is not None}
                self.facts.update(new_facts)
            except json.JSONDecodeError:
                print(f"⚠️ Fact extraction returned invalid JSON: {result_clean}")

        except Exception as e:
            print(f"⚠️ Fact extraction failed: {e}")

        # Semantic memory update
        if len(self.conversation_history) >= 2:
            snippet = f"User: {self.conversation_history[-2]['content']} | Assistant: {self.conversation_history[-1]['content']}"
            self.vector_db.add_texts([snippet])


    def retrieve_memory(self, query: str):
        """Combine structured facts + semantic recall."""
        fact_str = json.dumps(self.facts, indent=2)
        docs = self.vector_db.similarity_search(query, k=3)
        semantic_memories = [d.page_content for d in docs]
        return f"Facts:\n{fact_str}\nRelevant past: {semantic_memories}"


    def compress_history(self):
        """Summarize history but preserve facts."""
        summary = self.get_conversation_summary()
        fact_str = json.dumps(self.facts, indent=2)
        self.conversation_history = [
            {"role": "system", "content": f"Summary so far: {summary}\nKnown facts: {fact_str}", "timestamp": datetime.now().isoformat()}
        ]
        self.memory.clear()
        self.memory.chat_memory.add_ai_message(f"Summary: {summary}. Facts: {fact_str}")
        print("📝 Conversation compressed with hybrid memory.")


    # -----------------------------
    # Utilities
    # -----------------------------
    def get_conversation_summary(self) -> str:
        if not self.conversation_history:
            return "No conversation history available."
        summary_prompt = "Summarize this conversation briefly: " + json.dumps(self.conversation_history, indent=2)
        try:
            return self.llm.invoke(summary_prompt).content
        except Exception as e:
            return f"Unable to generate summary: {str(e)}"

    def clear_conversation_history(self):
        self.conversation_history = []
        self.memory.clear()
        self.facts = {}
        self.vector_db = FAISS.from_texts([], self.embeddings)
        print("🧹 All memory cleared.")


# -----------------------------
# main function
# -----------------------------
def main():
    API_KEY = os.getenv("GOOGLE_API_KEY")
    chatbot = HistoryAwareChatbot(api_key=API_KEY)
    print("🤖 Gemini Hybrid-Memory Chatbot with Persistence ready!")

    while True:
        user_input = input("\n👤 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "summary":
            print("\n📝 Summary:", chatbot.get_conversation_summary())
            continue
        elif user_input.lower() == "clear":
            chatbot.clear_conversation_history()
            continue
        elif user_input.lower() == "tokens":
            print(f"\n🔢 Tokens used so far: {chatbot.count_tokens()}")
            continue
        elif user_input.lower() == "facts":
            print("\n📌 Known facts:", json.dumps(chatbot.facts, indent=2))
            continue

        result = chatbot.chat(user_input)
        print(f"\n🤖 Assistant: {result['response']}")
        print(f"🔢 Tokens so far: {result.get('tokens_used', 'N/A')}")

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
