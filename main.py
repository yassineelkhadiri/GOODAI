"""Main entry point."""

from goodai.src.models import OpenAIModel, OpenSourceModel, Agent  # noqa:F401

OPEN_SOURCE_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
OPEN_AI_MODEL_NAME = "gpt-3.5-turbo"

# Please uncomment this line if you wish to disable logs.
# logging.disable(logging.CRITICAL)


def main() -> None:
    model = OpenSourceModel(OPEN_SOURCE_MODEL_NAME)
    # please add you api key if you wish to use this model.
    # model = OpenAIModel(OPEN_AI_MODEL_NAME)
    agent = Agent(model)

    print("Please enter NEW_SESSION if you wish to start a new session.")
    while True:
        user_input = str(input("User: "))
        if user_input == "NEW_SESSION":
            agent.new_session()
            print("New session created.")
        else:
            print(f"AI: {agent.interact(user_input)}")


if __name__ == "__main__":
    main()
