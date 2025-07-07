from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Sequence, Mapping
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from enum import Enum
from core.challenges import Pairing


class NarrativeState(BaseModel):
    # The story so far (excluding assessments and responses to assessments)
    story: List[AnyMessage] = Field(default_factory=list)
    survey_conversation: List[AnyMessage] = Field(default_factory=list)
    survey_data: str = Field(default_factory=str, description="Survey data about the child")
    finished_survey: bool = False
    # For challenge/narrative/assessment agent coordination
    next_triplet: Optional[Any] = None  # The current triplet to be used by the narrative agent
    used_triplets: Optional[list] = []  # List of all triplets that have been used
    user_responses: Optional[list] = []  # List of user responses to each triplet
    active_challenge: bool = False

class ChallengeState(BaseModel):
    messages: Sequence[BaseMessage] = Field(default_factory=list, description="History of messages")
    current_narrative_segment: List[AnyMessage] = Field(default_factory=list, description="The current segment of the story as decided by the manager")
    narrative_beat_info: Mapping[str, Any] = Field(default_factory=dict, description="Key information for the narrative beat")
    challenge_type: str = Field(default_factory=str, description="Which TILLS subtest to create a challenge for")
    modality: str = Field(default_factory=str, description="What kind of modality to use")
    story_history: List[AnyMessage] = Field(default_factory=list, description="The history of the story so far")
    challenge_history: list = Field(default_factory=list, description="The history of generated challenges") # A sequence of Challenges objects containing challenge information

# Todo: do we need to save the position of the established start (basal) and stop (ceiling) points?
class AssessmentState(BaseModel):
    basal: bool = Field(default=False, description="Whether the starting point of the task should be moved backwards or not")
    ceiling: bool = Field(default=False, description="Whether the stopping point has been reached or not")
    assessment_history: list = Field(default_factory=list, description="The history of evaluated student answers")

class FullState(BaseModel):
    # The full global state, namespaced per agent
    narrative: NarrativeState = Field(default_factory=NarrativeState)
    challenge: ChallengeState = Field(default_factory=ChallengeState)
    assessment: AssessmentState = Field(default_factory=AssessmentState)
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
    # For assessment_agent input
    student_response: Optional[str] = None
    # For assessment_agent output
    assessment_feedback: Optional[str] = None
