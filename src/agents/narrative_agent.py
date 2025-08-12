from langchain_core.messages import  AIMessage, SystemMessage, HumanMessage
from core.states import FullState
from .utils import BaseAgent
from core.config import survey_results as default_survey_results
from .prompts import NARRATIVE_PROMPTS

class NarrativeAgent(BaseAgent):
    def __init__(self, model, survey_results):
        super().__init__(name='Narrative Agent')

        self.model = model
        self.survey_prompt = NARRATIVE_PROMPTS['main_prompts']['survey_prompt']
        self.survey_format_prompt = NARRATIVE_PROMPTS['main_prompts']['survey_extract_data']
        self.prompt = NARRATIVE_PROMPTS['main_prompts']['narrative_prompt_template']
        self.challenge_prompts = NARRATIVE_PROMPTS['challenge_prompts']

    def __call__(self, state: FullState) -> FullState:
        """
        Generate a child-friendly story segment based on ongoing narrative and latest user input.

        Accepts and returns the global FullState, updating only the narrative namespace.
        """
        print("\n--- Running Narrative Agent ---")

        print(f"\nfinished_survey: {state.narrative.finished_survey!r}", end="\n\n")

        # If the survey is not finished, handle the survey logic
        if not state.narrative.finished_survey:
            return self.handle_survey(state)

        # Get current narrative history from the global state
        current_narrative = state.narrative.story
        print(f"\n\ncurrent_narrative: {current_narrative!r}\n")

        challenge_index = state.narrative.challenge_index

        if challenge_index is not None:
            next_challenge = state.challenge.challenge_history[challenge_index]
            print(f"\n[Narrative] next_challenge: {next_challenge}\n")

            print(f"\n\nnext_challenge: {next_challenge!r}\n")

            print("Incorporating challenge into the story...")
            if state.challenge.challenge_type == "Vocabulary Awareness":
                # Ensure next_challenge is a ChallengeTriplet object
                from core.challenges import ChallengeTriplet
                if next_challenge and isinstance(next_challenge, dict):
                    next_challenge = ChallengeTriplet.from_dict(next_challenge)

                challenge_prompt = self.challenge_prompts['vocabulary_awareness'].format(triplet=next_challenge.triplet)
            else:
                #TODO: Implement handling for other challenge types
                challenge_prompt = None

            story_segment = self.generate_story_segment(current_narrative, challenge_prompt=challenge_prompt, next_challenge=next_challenge)
        else:
            story_segment = self.generate_story_segment(current_narrative)
        story_segment = self.add_agent_metadata(story_segment)

        # Append AI turn
        state.narrative.story.append(story_segment)

        # Update full_history and last_agent as before
        state.full_history.append(story_segment)
        state.last_agent = self.name

        return state

    def handle_survey(self, state: FullState) -> FullState:
        """
        Handles the survey logic
        """
        # Check for skip command in latest user message
        if state.full_history and isinstance(state.full_history[-1], HumanMessage):
            user_msg = state.full_history[-1].content.strip()

            # Check for skip survey command
            if user_msg == "SKIP SURVEY":
                return self.finish_survey(state, skip_survey=True)

        # Append user message
        if isinstance(state.full_history[-1], HumanMessage):
            state.narrative.survey_conversation.append(state.full_history[-1])

        # Generate next survey question
        next_question = self.conduct_survey(state.narrative.survey_conversation)
        next_question = self.add_agent_metadata(next_question)

        # Append AI turn
        state.narrative.survey_conversation.append(next_question)

        # Update full_history and last_agent
        state.full_history.append(next_question)
        state.last_agent = self.name

        # Check if the survey is finished
        if next_question.content.endswith("<END>"):
            state = self.finish_survey(state)

        return state

    def conduct_survey(self, survey_conversation):
        print(f"\n--- Asking Survey Question ---")

        # Append system prompt
        messages = [SystemMessage(content=self.survey_prompt)] + survey_conversation

        next_question = self.model.invoke(messages)
        print(f"\nnext_question: {next_question!r}", end="\n\n")

        return next_question

    def finish_survey(self, state: FullState, skip_survey: bool = False):
        """
        Finish the survey and prepare the story to begin.
        """
        print("[NarrativeAgent] Finishing survey...")

        # Set finished_survey to True
        state.narrative.finished_survey = True

        # If skip_survey is True, use default survey results
        if skip_survey:
            print("[NarrativeAgent] Skipping survey and using default survey results.")
            state.narrative.survey_data = str(default_survey_results)
        else:
            # Format the survey results
            state.narrative.survey_data = self.format_survey_results(state.narrative.survey_conversation)

        self.prompt = self.prompt.format(survey_results=state.narrative.survey_data)

        # Reset the story to start fresh
        start_message = AIMessage(content="**--- BEGINNING STORY ---**\n---\n")
        start_message = self.add_agent_metadata(start_message)
        state.narrative.story = [start_message]

        # Update full history and last agent
        state.full_history.append(start_message)
        state.last_agent = self.name

        return state

    def generate_story_segment(self, current_narrative, challenge_prompt=None, next_challenge=None):
        """
        Generate a story segment based on the current narrative and (optionally) a challenge prompt.
        """
        print(f"\n--- Generating Story Segment ---")

        # Append system prompt
        if challenge_prompt:
            prompt = challenge_prompt
        else:
            prompt = self.prompt
        messages = [SystemMessage(content=prompt)] + current_narrative

        # print(f"\nmessages: {messages!r}", end="\n\n")

        story_segment = self.model.invoke(messages)
        # print(f"\nGenerated story segment: {story_segment.content}", end="\n\n")

        print(f"\n[Narrative] challenge_prompt: {challenge_prompt}\n")
        print(f"\n[Narrative] next_challenge: {next_challenge}\n")

        # If this is a challenge, add the triplet as metadata
        if challenge_prompt and next_challenge:
            if getattr(story_segment, 'metadata', None) is None:
                story_segment.metadata = {}
            story_segment.metadata['challenge_triplet'] = next_challenge
        return story_segment

    def add_agent_metadata(self, message):
        if isinstance(message, AIMessage):
            if getattr(message, 'metadata', None) is None:
                message.metadata = {}
            message.metadata["agent"] = self.name
            return message

    def format_survey_results(self, survey_conversation):
        """
        Format the survey results into a dictionary.
        """
        print(f"\n--- Formatting Survey Results ---")

        # Convert the survey conversation into a string
        survey_conversation_string = ""
        for msg in survey_conversation:
            if isinstance(msg, HumanMessage):
                survey_conversation_string += (f"\nUser: {msg.content!r}")
            elif isinstance(msg, AIMessage):
                survey_conversation_string += (f"\nAI: {msg.content!r}")

        # Append system prompt
        messages = [SystemMessage(content=self.survey_format_prompt), HumanMessage(content=f"Here is the conversation for you to turn into a dictionary of survey data collected on the user:\n\n{survey_conversation_string}")]

        survey_data = self.model.invoke(messages)
        print(f"\nSurvey data: {survey_data.content}", end="\n\n")

        return survey_data.content.strip()
