from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, Any, Mapping
from core import survey_results
from core.states import FullState
from agents.utils import BaseAgent

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


class NarrativeAgent(BaseAgent):
    def __init__(self, model, survey_results):
        super().__init__(name='Narrative Agent')

        self.model = model
        self.prompt = NARRATIVE_PROMPT_TEMPLATE.format(**survey_results)

    def __call__(self, state: FullState) -> FullState:
        """
        Generate a child-friendly story segment based on ongoing narrative and latest user input.

        Accepts and returns the global FullState, updating only the narrative namespace.
        """
        print("\n--- Running Narrative Agent ---")

        # Get current narrative history from the global state
        current_narrative = state.narrative.story

        print()
        print(f'current_narrative: {current_narrative!r}')
        print()

        # Generate the story segment
        story_segment = self.generate_story_segment(current_narrative)

        # Add agent metadata before appending
        if isinstance(story_segment, AIMessage):
            story_segment.metadata = dict(story_segment.response_metadata or {})
            story_segment.metadata["agent"] = self.name

        # Only append the story_segment here; HumanMessages will be handled elsewhere
        state.narrative.story.append(story_segment)

        # Update full_history and last_agent as before
        state.full_history.append(story_segment)
        state.last_agent = self.name

        return state

    def generate_story_segment(self, current_narrative):
        """
        Generate a story segment based on the current narrative.
        """
        print(f"\n--- Generating Story Segment ---")

        # converted = []
        # for narrative in current_narrative:
        #     converted.append(SystemMessage(content=narrative.content)
        #                      if isinstance(narrative, BaseMessage) else narrative)


        messages = [SystemMessage(content=self.prompt)] + current_narrative

        print()
        print(f"messages: {messages!r}")
        print()

        story_segment = self.model.invoke(messages)
        print()
        print(f"Generated story segment: {story_segment.content}")
        print()

        return story_segment

    def ensure_age_appropriate(self, story_segment):
        print(f"\n--- Ensuring Age Appropriateness ---")
        # Implement logic to check if the story_segment is appropriate for children
        return story_segment
