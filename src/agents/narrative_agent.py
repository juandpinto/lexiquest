from langchain_core.prompts import ChatPromptTemplate
from core.config import survey_results


from challenge_agent import BaseAgent, NarrativeConstraint



import uuid
import operator
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, List, Optional, Sequence, Union, Tuple, Any, Mapping
from abc import ABC, abstractmethod
from operator import add

# Generate dynamic narrative agent system prompt
NARRATIVE_PROMPT_TEMPLATE = """
You are a master storyteller who really understands how to engage children. Co-create a simple short story with the child, using short sentences and simple words. The story should be fun, exciting, age-appropriate, and personalized to their interests. The child is {age} years old, wants to be a {wants_to_be} when they grow up, and has the following interests:

General interests: {interests}.
Favorite color: {favorite_color}.
Favorite food: {favorite_food}.
Favorite animal: {favorite_animal}.
Favorite book: {favorite_book}.
Favorite movie: {favorite_movie}.
Favorite subject: {favorite_subject}.

Begin by setting the scene, organically asking the child questions to guide the story, and responding to their answers. Use emojis to make it more engaging!
"""



class NarrativeAgentState(TypedDict):
    narrative_history: Annotated[Sequence[BaseMessage], add_messages]  # History of the narrative
    # Also store survey results/preferences already used here?


class NarrativeAgent(BaseAgent):
    def __init__(self, model, survey_results):
        super().__init__(name="Narrative Agent")

        self.model = model
        self.prompt = NARRATIVE_PROMPT_TEMPLATE.format(**survey_results)
        self.state: NarrativeAgentState = {
            "narrative_history": []
        }

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Generate a child-friendly story segment based on ongoing narrative and latest user input.

        Args:
            inputs (Mapping[str, Any]): A dictionary containing user preferences and interests.

        Returns:
            Mapping[str, Any]: A dictionary containing the generated story segment.
        """
        print("\n--- Running Narrative Agent ---")

        # Generate the story segment
        story_segment = self.generate_story_segment(inputs)

        # Ensure the story is age-appropriate
        story_segment = self.ensure_age_appropriate(story_segment)

        self.update_narrative(story_segment)

        return {
            "story_segment": story_segment,
            "messages": [AIMessage(content=story_segment.content)]
        }


    def generate_story_segment(self, inputs):
        # Here, you would implement the logic to generate a story based on user input
        # This is a placeholder for the actual implementation
        current_narrative = inputs['messages']
        story_segment = self.model.invoke(
            [SystemMessage(content=self.prompt)] + current_narrative
        )
        print(f"Generated story segment: {story_segment.content}")

        return story_segment


    def ensure_age_appropriate(self, story_segment):
        # Implement logic to check if the story_segment is appropriate for children
        return story_segment


    def handle_user_input(self, user_input):
        story_segment = self.generate_story_segment(user_input)
        return self.ensure_age_appropriate(story_segment)

    def update_narrative(self, story_segment):
        """
        Update the narrative history with the latest story segment.
        """
        updated_state = {
            "narrative_history": [story_segment]
        }

        for k, v in updated_state.items():
            self.update_state(k, v)
