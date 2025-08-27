import json
import os 
from dotenv import load_dotenv
import logging

from HistoryAwareChatbot import HistoryAwareChatbot

load_dotenv(override=True)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 
def main():
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        print("❌ Please set GOOGLE_API_KEY in your .env file")
        return

    try:
        chatbot = HistoryAwareChatbot(api_key=API_KEY, max_tokens=300)
        print("🤖 Enhanced Gemini Hybrid-Memory Chatbot ready!")
        print("Commands: 'quit', 'summary', 'clear', 'tokens', 'facts', 'stats'")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return

    while True:
        try:
            # Take input from user
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue

            command = user_input.lower()

            # Commands handling
            if command == "quit":
                break
            elif command == "summary":
                print(f"\n📝 Summary: {chatbot.get_conversation_summary()}")
                continue
            elif command == "clear":
                chatbot.clear_conversation_history()
                print("🧹 Memory cleared!")
                continue
            elif command == "tokens":
                print(f"\n🔢 Current tokens: {chatbot.count_tokens()}")
                continue
            elif command == "facts":
                if chatbot.facts:
                    print(f"\n📌 Known facts: {json.dumps(chatbot.facts, indent=2)}")
                else:
                    print("\n📌 No facts recorded yet.")
                continue
            elif command == "stats":
                stats = chatbot.get_memory_stats()
                print(f"\n📊 Memory Stats: {json.dumps(stats, indent=2)}")
                continue

            # Regular chat
            result = chatbot.chat(user_input)

            # Print assistant response only
            print(f"\n🤖 Assistant: {result['response']}")

            # Optional: show high token warning
            if result.get("tokens_used", 0) > 200:
                print(f"⚠️  High token usage: {result.get('tokens_used')}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            print("❌ An error occurred. Please try again.")

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()