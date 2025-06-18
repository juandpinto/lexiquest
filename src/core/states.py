from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class NarrativeState(BaseModel):
    # The story so far (excluding assessments and responses to assessments)
    story: List[BaseMessage] = Field(default_factory=list)

class ChallengeState(BaseModel):
    assessments: List[Dict[str, Any]] = Field(default_factory=list)
    responses: List[Dict[str, Any]] = Field(default_factory=list)
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
