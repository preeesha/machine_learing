import json
from datetime import datetime
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class HistoryAwareChatbot:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

        # Configure Google GenAI client (for token counting)
        genai.configure(api_key=api_key)
        self.client = genai

        # Use Gemini via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048,
        )

        # LangChain memory: remembers all interactions
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        # Combine LLM + Memory for a conversational chain
        self.chain = ConversationChain(llm=self.llm, memory=self.memory)

        self.conversation_history = []

    def compress_history(self):
        
        summary = self.get_conversation_summary()

        # preserve important facts (like user's name) explicitly
        facts = "Known facts: The user's name is Prisha."

        self.conversation_history = [
            {"role": "system", "content": f"{summary}\n\n{facts}", "timestamp": datetime.now().isoformat()}
        ]
        self.memory.clear()
        self.memory.chat_memory.add_ai_message(f"Summary: {summary}. {facts}")
        print("📝 Conversation compressed into summary with user facts.")

        
    def chat(self, user_input: str) -> dict:
        timestamp = datetime.now().isoformat()
        self.conversation_history.append(
            {"role": "user", "content": user_input, "timestamp": timestamp}
        )
        
         # 🔥 Check token usage before responding
        if self.count_tokens() > 400:
            print("⚠️ Token limit exceeded. Compressing history...")
            self.compress_history()
            # re-add the latest user input
            self.conversation_history.append(
                {"role": "user", "content": user_input, "timestamp": timestamp}
            )
            print(self.conversation_history)
        try:
            response = self.chain.run(user_input)
            self.conversation_history.append(
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            )
            return {
                "response": response,
                "conversation_length": len(self.conversation_history),
                "timestamp": timestamp,
                "tokens_used": self.count_tokens(),   # NEW: show token usage
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

    def count_tokens(self) -> int:
        """Count tokens used so far in the conversation"""
        try:
            contents = [
                {"role": h["role"], "parts": [h["content"]]}
                for h in self.conversation_history
            ]
            model = self.client.GenerativeModel(self.model_name)
            count = model.count_tokens(
                contents=contents
            )
            return count.total_tokens
        except Exception as e:
            print(f"⚠️ Token count failed: {e}")
            return -1

    def get_conversation_summary(self) -> str:
        if not self.conversation_history:
            return "No conversation history available."
        summary_prompt = "Summarize this conversation: " + json.dumps(
            self.conversation_history, indent=2
        )
        try:
            # Direct summary from Gemini
            return self.llm.invoke(summary_prompt).content
        except Exception as e:
            return f"Unable to generate summary: {str(e)}"

    def clear_conversation_history(self):
        self.conversation_history = []
        self.memory.clear()
        print("Conversation history cleared")


def main():
    API_KEY = "AIzaSyBZcfAQI1fxLebXl2aBw4yRWeUhDRK0p6E"  # Paste your key here!
    chatbot = HistoryAwareChatbot(api_key=API_KEY)
    print("🤖 Gemini History-Aware Chatbot ready!")
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
        result = chatbot.chat(user_input)
        print(f"\n🤖 Assistant: {result['response']}")
        print(f"🔢 Tokens so far: {result['tokens_used']}")
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
