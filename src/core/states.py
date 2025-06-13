from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

# --- Agent-specific namespaces ---

class NarrativeState(BaseModel):
    # The story so far (excluding assessments and responses to assessments)
    story: List[BaseMessage] = Field(default_factory=list)

class ChallengeState(BaseModel):
    # All assessments administered so far and user responses
    assessments: List[Dict[str, Any]] = Field(default_factory=list)
    responses: List[Dict[str, Any]] = Field(default_factory=list)
    # Optionally, you can add a challenge_history or use Pydantic models for more structure
    # challenge_history: List[ChallengeTriplet] = Field(default_factory=list)

class FullState(BaseModel):
    # The full global state, namespaced per agent
    narrative: NarrativeState = Field(default_factory=NarrativeState)
    challenge: ChallengeState = Field(default_factory=ChallengeState)
    # Optionally, include the full conversation and any other metadata
    full_history: List[str] = Field(default_factory=list)
    # Example: the last agent to produce output
    last_agent: Optional[str] = None
    # Store the manager's routing decision for the router node
    manager_decision: Optional[Dict[str, Any]] = None
    # You can add more shared/global fields as needed

# --- Example of agent node functions ---

def narrative_agent_node(state: FullState, story_segment: BaseMessage) -> FullState:
    # Update only the narrative namespace
    state.narrative.story.append(story_segment)
    state.last_agent = "narrative"
    state.full_history.append(story_segment.content)
    return state

def challenge_agent_node(state: FullState, assessment: Dict[str, Any], response: Dict[str, Any]) -> FullState:
    # Update only the challenge namespace
    state.challenge.assessments.append(assessment)
    state.challenge.responses.append(response)
    state.last_agent = "challenge"
    state.full_history.append(f"Assessment: {assessment.get('question', '')} | Response: {response.get('response', '')}")
    return state

def manager_node(state: FullState, manager_decision: Dict[str, Any]) -> FullState:
    # Store the manager's decision in the state for the router node
    state.manager_decision = manager_decision
    state.last_agent = "manager"
    state.full_history.append(f"Manager decision: {manager_decision}")
    return state

# Example of how this might be used in a workflow:
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage

    state = FullState()
    user_inputs = [
        HumanMessage(content="Once upon a time..."),
        HumanMessage(content="He found a treasure."),
        HumanMessage(content="But then a dragon appeared!")
    ]
    for inp in user_inputs:
        # Simulate manager decision
        manager_decision = {"next_agent": "narrative_agent", "task": "Continue the story"}
        state = manager_node(state, manager_decision)
        if state.manager_decision["next_agent"] == "narrative_agent":
            state = narrative_agent_node(state, inp)
        elif state.manager_decision["next_agent"] == "challenge_agent":
            assessment = {"question": "What happened to the main character?", "turn": len(state.challenge.assessments)}
            response = {"response": inp.content, "turn": len(state.challenge.assessments)}
            state = challenge_agent_node(state, assessment, response)
        print(state.json(indent=2))
