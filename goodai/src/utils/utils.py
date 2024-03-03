from tqdm import tqdm

from goodai.src.models import Agent

random_strings = [
    "The quick brown fox jumps over the lazy dog.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "In the beginning God created the heavens and the earth.",
    "To be, or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "You don't have to be great to start, but you have to start to be great.",
    "The only thing we have to fear is fear itself.",
    "To infinity and beyond!",
    "May the Force be with you.",
    "Life is like a box of chocolates. You never know what you're gonna get.",
    "Say hello to my little friend!",
    "Elementary, my dear Watson.",
]


def mock_interactions(agent: Agent) -> None:
    """
    This function is used to mimic a normal conversation
    with the agent using some random inputs.
    """
    for string in tqdm(random_strings):
        agent.interact(string)
