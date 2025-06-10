from langchain_core.prompts import ChatPromptTemplate
from core.config import survey_results

# Generate dynamic narrative agent system prompt
narrative_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""You are a master storyteller who really understands how to engage children. Co-create a simple short story with the child, using short sentences and simple words. The story should be fun, exciting, age-appropriate, and personalized to their interests. The child is {age} years old, wants to be a {wants_to_be} when they grow up, and has the following interests:

General interests: {interests}.
Favorite color: {favorite_color}.
Favorite food: {favorite_food}.
Favorite animal: {favorite_animal}.
Favorite book: {favorite_book}.
Favorite movie: {favorite_movie}.
Favorite subject: {favorite_subject}.

Begin by setting the scene, organically asking the child questions to guide the story, and responding to their answers. Use emojis to make it more engaging!""",
        ),
        # MessagesPlaceholder(variable_name="messages"), # You might need to add MessagesPlaceholder from langchain_core.prompts if you use it
    ]
)

# Formatted prompt ready for use
formatted_narrative_prompt = narrative_prompt_template.format(**survey_results)

class NarrativeAgent:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def generate_story(self, user_input):
        # Here, you would implement the logic to generate a story based on user input
        # This is a placeholder for the actual implementation
        story = f"Once upon a time, in a land far away, there was a child who loved {user_input['interests']}. "
        story += f"They dreamed of becoming a {user_input['wants_to_be']} and had many adventures!"
        return story

    def ensure_age_appropriate(self, story):
        # Implement logic to check if the story is appropriate for children
        # This is a placeholder for the actual implementation
        return story  # Assuming the story is appropriate for now

    def handle_user_input(self, user_input):
        story = self.generate_story(user_input)
        return self.ensure_age_appropriate(story)
