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
    "I have a dream.",
    "May the Force be with you.",
    "Life is like a box of chocolates. You never know what you're gonna get.",
    "Let's go for a ride",
    "Say hello to my little friend!",
    "Elementary, my dear Watson.",
    "Toto, I've a feeling we're not in Kansas anymore.",
    "E.T. phone home.",
    "Here's looking at you, kid.",
    "I'll be back.",
    "I see dead people.",
    "Hasta la vista, baby.",
    "Don't you dare lie to me",
    "There's no place like home.",
    "You talking to me?",
    "Go ahead, make my day.",
    "Keep your friends close, but your enemies closer.",
    "Just keep swimming.",
    "I'm the king of the world!",
    "I'll have what she's having.",
    "You had me at hello.",
    "I am your father.",
    "I need to change my keyboard",
    "I'm gonna make him an offer he can't refuse.",
    "I feel the need... the need for speed!",
    "There's no crying in baseball!",
    "I need to see a doctor",
    "The weather is great today",
    "Houston, we have a problem.",
    "I like to play football",
    "You can't handle the truth!",
    "You're gonna need a bigger boat.",
    "I want some pasta",
    "You look nice today",
    "I want to travel",
    "I am more of a cats person",
    "Frankly, my dear, I don't give a damn.",
    "See you outside",
    "Here's Johnny!",
]


def mock_interactions(agent: Agent) -> None:
    """
    This function is used to mimic a normal conversation
    with the agent using some random inputs.
    """
    for string in tqdm(random_strings):
        agent.interact(string)
