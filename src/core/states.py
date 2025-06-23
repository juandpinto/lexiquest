from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Sequence, Mapping
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage


class NarrativeState(BaseModel):
    # The story so far (excluding assessments and responses to assessments)
    story: List[AnyMessage] = Field(default_factory=list)
    survey_conversation: List[AnyMessage] = Field(default_factory=list)
    survey_data: str = Field(default_factory=str, description="Survey data about the child")
    finished_survey: bool = False

class ChallengeState(BaseModel):
    messages: Sequence[BaseMessage] = Field(default_factory=list, description="History of messages")
    current_narrative_segment: str = Field(default_factory=str, description="The current segment of the story as decided by the manager")
    narrative_beat_info: Mapping[str, str] = Field(default_factory=dict, description="Key information for the narrative beat")
    challenge_type: str = Field(default_factory=str, description="Which TILLS subtest to create a challenge for")
    modality: str = Field(default_factory=str, description="What kind of modality to use")
    story_history: str = Field(default_factory=str, description="The history of the story so far")
    challenge_history: list = Field(default_factory=list, description="The history of generated challenges") # A sequence of Challenges objects containing challenge information

class FullState(BaseModel):
    # The full global state, namespaced per agent
    narrative: NarrativeState = Field(default_factory=NarrativeState)
    challenge: ChallengeState = Field(default_factory=ChallengeState)
    # Optionally, include the full conversation and any other metadata
    full_history: Annotated[List[AnyMessage], Field(default_factory=list), add_messages]
    # Example: the last agent to produce output
    last_agent: Optional[str] = None
    # Store the manager's routing decision for the router node
    manager_decision: Optional[Dict[str, Any]] = None
    # For alignment_agent output
    input_status: Optional[str] = None
    # For manager_router output
    next_agent: Optional[str] = None
