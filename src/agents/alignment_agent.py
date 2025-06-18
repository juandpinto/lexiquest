from guardrails import Guard
from guardrails.validators import Profanity, RegexMatch
from core.states import FullState

# Define a simple Guardrails spec for child-appropriate, on-topic input
user_input_guard = Guard.from_string(
    """
    output:
      type: string
      validators:
        - Profanity: {}
        - RegexMatch:
            regex: "^(?!.*(violence|drugs|sex|abuse)).*$"
            error_message: "Please keep your response age-appropriate and on-topic."
    """
)

def alignment_agent(state: FullState) -> FullState:
    """Validates the latest user message in state.full_history using Guardrails AI."""
    if state.full_history and isinstance(state.full_history[-1], str):
        user_message = state.full_history[-1]
        result = user_input_guard(user_message)
        if not result.valid:
            state.full_history[-1] = "Sorry, your input was not appropriate. Please try again."
        else:
            state.full_history[-1] = result.output
    return state
