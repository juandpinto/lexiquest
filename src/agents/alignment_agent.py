from guardrails.hub import ProfanityFree
from core.states import FullState
from agents.utils import BaseAgent
from langchain_core.messages import HumanMessage, AIMessage
from pprint import pprint

# Set up the ProfanityFree validator
profanity_validator = ProfanityFree(on_fail="exception")

class AlignmentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='Alignment Agent')
        self.validator = profanity_validator

    def __call__(self, state: FullState) -> FullState:
        """
        Validates the latest user message in state.full_history using Guardrails AI.
        Accepts and returns the global FullState, updating only the relevant namespaces.
        """
        print("\n--- Running Alignment Agent ---")

        if state.full_history and isinstance(state.full_history[-1], HumanMessage):
            user_message = state.full_history[-1].content
            try:
                result = self.validator(user_message)
                print(f"Input: {user_message!r} | Valid: True")
                state.input_status = "valid_input"
                # Also append to story if valid
                state.narrative.story.append(state.full_history[-1])
            except Exception as e:
                print(f"Input: {user_message!r} | Valid: False | Exception: {e}")
                state.full_history.append(AIMessage(content="Sorry, your input was not appropriate. Please try again."))
                state.input_status = "invalid_input"
        else:
            state.input_status = "valid_input"

        # Save the state to a file for debugging purposes
        state.save_to_file("state.json")
        return state
