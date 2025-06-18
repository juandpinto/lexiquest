from guardrails.hub import ProfanityFree
from core.states import FullState

# Set up the ProfanityFree validator
profanity_validator = ProfanityFree(on_fail="exception")

def alignment_agent(state: FullState) -> FullState:
    """Validates the latest user message in state.full_history using Guardrails AI."""
    if state.full_history and isinstance(state.full_history[-1], str):
        user_message = state.full_history[-1]
        try:
            result = profanity_validator(user_message)
            if not result.valid:
                state.full_history[-1] = "Sorry, your input was not appropriate. Please try again."
                state.input_status = "invalid_input"
            else:
                state.full_history[-1] = user_message
                state.input_status = "valid_input"
        except Exception:
            state.full_history[-1] = "Sorry, your input was not appropriate. Please try again."
            state.input_status = "invalid_input"
    else:
        state.input_status = "valid_input"
    return state
