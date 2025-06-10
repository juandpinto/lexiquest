import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Configuration settings
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEFAULT_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.8

    @staticmethod
    def validate_keys():
        if not Config.OPENAI_API_KEY and not Config.GOOGLE_API_KEY:
            raise ValueError("No API key found. Please set OPENAI_API_KEY or GOOGLE_API_KEY in the environment variables.")

# Sample survey results
survey_results = {
    'age': 7,
    'interests': ['space', 'dinosaurs'],
    'wants_to_be': 'astronaut',
    'favorite_color': 'blue',
    'favorite_food': 'pizza',
    'favorite_animal': 'dolphin',
    'favorite_book': 'Harry Potter',
    'favorite_movie': 'Toy Story',
    'favorite_subject': 'science',
}
