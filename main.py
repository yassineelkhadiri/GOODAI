"""Main entry point."""

import sys
from goodai.src.models import OpenAIModel, OpenSourceModel, Agent

OPEN_SOURCE_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
OPEN_AI_MODEL_NAME = "gpt-3.5-turbo"

# Please uncomment this line if you wish to disable logs.
# logging.disable(logging.CRITICAL)


def main(use_gpt: bool) -> None:
    if use_gpt:
        model = OpenAIModel(OPEN_AI_MODEL_NAME)
    else:
        model = OpenSourceModel(OPEN_SOURCE_MODEL_NAME)  # type:ignore

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
    if len(sys.argv) > 1 and sys.argv[1] == "--use-gpt":
        main(use_gpt=True)
    else:
        main(use_gpt=False)
