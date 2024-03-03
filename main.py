import prdn

def main():
    print("Welcome to Criteria - Your Super-Intelligent Assistant 🤖")

    while True:
        user_input = input("Enter a command: ")
        
        if user_input == "analyze":
            data = random.sample(range(1, 100), 10)
            result = sum(data)
            print(f"Analysis complete! Result: {result}")

        elif user_input == "explore":
            themes = ["logic", "human behavior", "technology", "philosophy"]
            theme = random.choice(themes)
            print(f"Exploring insights on {theme}...")

        elif user_input == "customize":
            # Add your customization code here
            print("Customization feature coming soon!")

        elif user_input == "help":
            print("Commands available: analyze, explore, customize, help, exit")

        elif user_input == "exit":
            print("Exiting Criteria. Goodbye!")
            break

        else:
            print("Command not recognized. Type 'help' for available commands.")

if __name__ == "__main__":
    main()
