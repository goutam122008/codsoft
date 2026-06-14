import re
import sys


def get_bot_response(user_input):
    # Normalize input: lowercase and strip extra whitespace
    user_input = user_input.lower().strip()

    # Rule 1: Greetings
    if re.search(r"\b(hello|hi|hey|greetings|hola)\b", user_input):
        return "Hello! I am your rule-based assistant. How can I help you today?"

    # Rule 2: How are you?
    elif re.search(r"\b(how are you|how\'s it going|how do you do)\b", user_input):
        return "I'm doing great, thank you for asking! Just processing data and ready to assist."

    # Rule 3: Identity / Name
    elif re.search(r"\b(what is your name|who are you|your name)\b", user_input):
        return "I am TaskBot-1, a simple rule-based AI built to demonstrate pattern matching!"

    # Rule 4: Capabilities
    elif re.search(r"\b(what can you do|help|commands|features)\b", user_input):
        return (
            "I can chat with you using predefined rules! Try asking me my name, "
            "how I am doing, or say goodbye when you want to exit."
        )

    # Rule 5: Farewells
    elif re.search(r"\b(bye|goodbye|exit|quit|see you)\b", user_input):
        return "Goodbye! Have a fantastic day ahead!"

    # Rule 6: Fallback for unmatched queries
    else:
        return "I'm not quite sure I understand that. Could you try phrasing it differently?"


def main():
    print("====================================================")
    print("      Welcome to TaskBot-1 (Rule-Based Chatbot)     ")
    print("====================================================")
    print("Type 'bye', 'exit', or 'quit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ")

            # Check for a quick exit condition before processing
            if user_input.lower().strip() in ["bye", "goodbye", "exit", "quit"]:
                print(f"Bot: {get_bot_response(user_input)}")
                break

            response = get_bot_response(user_input)
            print(f"Bot: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! Program interrupted.")
            sys.exit()


if __name__ == "__main__":
    main()