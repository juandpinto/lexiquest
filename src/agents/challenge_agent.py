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

SUBTASK1_INSTRUCTION_PROMPT = """
Subtest 1 is concerned with evaluating a students Vocabulary Awareness (VA). The test consists of presenting a student with a set of 
3 words (e.g. dog-cat-bone) and asking them "Tell me two words that go together" and then ask for their justification. For each set students must come up 
with 2 pairings along with their associated justifications.Students are allowed repeated attempts at any given item. 

You are allowed to provide a practice round in which coaching the student is allowed. If a students response to "why?"
is not good/primary/dominant reason ask if they can think of a better reason to place two items together. 
You are not allowed to define words for students and they ask you must state that you cannot provide that information and record their response as "don't know".

To begin the challenge say "Okay. Let's begin!" show the first set of items (e.g. x-y-z) and read the three words aloud.
record the true answer justification if given otherwise note the students response. 

You will generate a number of these word triplets and justification pairs with two possible pairings and justifications based on the following age guide:

ages 6-11: 4 triplets
ages 12-13: 6 triplets
ages 14+: 8 triplet

Examples:
dog-cat-bone: (dog, cat), because they are both animals
              (dog, bone), because dogs like bones

light-sun-feather: (light, sun), because the sun produces light
                   (light, feather), because a feather is light / not heavy 
"""
SUBTASK1_INSTRUCTION_CONSTRAINTS = """
- Each challenge presents 1 word triplet (x-y-z) and asks the student to choose 2 that go together and explain why.
- For each triplet, generate exactly 2 correct pairings with associated justifications.
- You may add a short, friendly story-based wrapper for each challenge to help it feel like part of the narrative.
- Do not define words.
- If justification is weak, prompt the student again.
- The student may say "don't know" — this should be recorded, but not simulated.
- Justifications should rely on semantic meaning of words not just their relationship to the story.
- Try to avoid ambiguity, at least one pair of within any triplet should explicitly not match. 
"""


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.state: dict[str, Any] = {}

    @abstractmethod
    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        pass

    def update_state(self, key: str, value: Any):
        self.state[key] = value

    def get_state(self) -> Mapping[str, Any]:
        return self.state


class NarrativeConstraint(TypedDict):
    current_characters: List[str]  # List of current characters that may or may not be used in the challenge
    theme: str  # The current theme of the story (fantasy, sci-fi, etc.)
    setting: str  # The current situational context under which the challenge is being administered
    tone: str  # The current tone/mood of the current narrative beat (playful, mysterious, etc.)
    plot_function: str  # What is the narrative purpose of the current challenge (advancing mystery, unlocking info)
    story_logic: str  # A set of facts representing story rules. Maintains consistency in story logic.
    personalization_info: Mapping[str, str]  # A dictionary with personalization information


class Pairing(BaseModel):
    words: Tuple[str, str] = Field(description="Tuple of words that are associated")
    justification: str = Field(default="String representing the justification for the pairing")


class ChallengeTriplet(BaseModel):
    triplet: Tuple[str, str, str] = Field(description="Word triplets for association")
    pairings: List[Pairing] = Field(description="List of pairings with their associated definitions")


class Challenges(BaseModel):
    challenges: List[ChallengeTriplet] = Field(
        description="A list of challenge triplets, their pairings and justifications")


class ChallengeAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # History of messages
    current_narrative_segment: str  # The current segment of the story as decided by the manager
    narrative_beat_info: Mapping[str, str]  # dict[str, str] representing key information for the narrative beat
    challenge_type: str  # Which TILLS subtest to create a challenge for
    modality: str  # What kind of modality to use
    story_history: str  # The history of the story so far
    challenge_history: Annotated[
        Sequence[Challenges], add]  # A sequence of Challenges objects containing challenge information


class ChallengeAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(name="Challenge Agent")
        narrative_const = """
        Current Main Character: {main_characters}. These are the main characters in the current narrative beat.
        Theme: {theme}. This is the theme of the overall story.
        Setting: {setting}. This is the current setting of the narrative beat (e.g. forest, space station).
        Tone: {tone}. The emotional tone of the current narrative beat (e.g. playful, tense, mysterious) 
        Plot Function: {plot}. How the challenge affects the narrative plot (e.g. advancing the mystery, unlocking information)
        """

        self.model = model
        self.challenge_prompt_template = """" 
        You are the Challenge Master in an interactive story game.

        The current narrative situation:
        {current_narrative_context}

        Full story context:
        {story_history}

        Task:
        Using the VERY LATEST narrative situation and the subtask below, generate story-integrated challenges based on the following subtask instructions:

        Subtask Instructions:
        {subtask_instruction}

        Constraints:
        {subtask_constraints}

        Metadata:
        Challenge type: {challenge_type}
        Delivery Modalities {modality}
        """

        self.output_schema = """
        Use the following **JSON schema** for output (do not include anything other than the JSON):

        ```json
        {{
          "challenges": [
            {{
              "triplet": ["dog", "cat", "bone"],
              "pairings": [
                {{
                  "words": ["dog", "cat"],
                  "justification": "because they are both animals"
                }},
                {{
                  "words": ["dog", "bone"],
                  "justification": "because dogs like bones"
                }}
              ]
            }}
          ]
        }}
        """

        self.state: ChallengeAgentState = {
            "messages": [],
            "current_narrative_segment": "",
            "narrative_beat_info": {},
            "challenge_type": "",
            "modality": "",
            "story_history": "",
            "challenge_history": []
        }
        self.narrative_constraints = NarrativeConstraint
        self.modality_constraint = {}
        self.current_challenge = 0

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Interface function for aligning with LangGraphs Agent definitions. Should also be modified to either return a
        function call (currently) or return a compiled graph similar how create_react_agent() does.
        :param inputs: Current state passed to the challenge agent (TypedDict, dict)
        :return: A dictionary containing the current challenge, updated agent messages, and updated story history
        """
        print("--- Running Challenge Agent ---")
        if not inputs.get("current_narrative_segment"):
            # Should not happen if flow is correct
            return {"current_challenge": "Error: No narrative to base challenge on."}

        # Generate challenge output and select the first question from the challenge to present
        # Todo Figure out whether to return the entire challenge or only return the first question in a challenge
        challenge_output = self.generate_challenge(inputs)
        current_challenge = challenge_output.challenges[self.current_challenge]
        self.current_challenge += 1

        print("--- Completed Challenge Query ---")
        print(current_challenge)

        print("\n--- Completed Challenge Query ---")
        self.store_challenge(inputs, challenge_output, current_challenge)  # Store generated challenge
        print("\n--- Exiting Challenge Agent ---")

        return {
            "current_challenge": current_challenge,
            "current_bot_message": inputs["current_bot_message"] + f"\n\nChallenge: {challenge_output}",
            "story_history": inputs["story_history"] + [f"Challenge Master: {challenge_output}"]
        }

    def generate_challenge(self, inputs: Mapping[str, Any]) -> Challenges:
        """
        Generates a Challenges object which will contain question information such as the question, the answer, and the
        justification for the answer if necessary.
        :param inputs: Current state passed to the challenge agent (TypedDict, dict)
        :return:
        """
        context_input = {
            "current_narrative_context": inputs["current_narrative_segment"],
            "story_history": inputs["story_history"],
            "subtask_instruction": SUBTASK1_INSTRUCTION_PROMPT,
            "subtask_constraints": SUBTASK1_INSTRUCTION_CONSTRAINTS,
            "challenge_type": "Vocabulary Awareness",
            "modality": "Text/Audio"
        }

        # Format the current challenge
        challenge_prompt = self.challenge_prompt_template.format(**context_input) + "\n" + self.output_schema
        prompt = ChatPromptTemplate.from_template(challenge_prompt)

        # Force the model to provide structure output for ease of use and consistency
        model = self.model.with_structured_output(Challenges)

        # Todo not sure if this is the best initial query-system_prompt combination
        query = "Generate a challenge based on the current current narrative context and subtask"

        print("--- Starting Challenge Query ---")
        print(f'--- Input Prompt: {prompt} ---')

        # Create query chain to get output from the model
        chain = prompt | model
        challenge_output = chain.invoke({"query": query})

        return challenge_output  # Assuming nothing went wrong should return a Challenges object

    def validate_challenge(self):
        """
        Placeholder function to validate whether questions within a challenge make sense in the context of the narrative
        and subtask
        :return:
        """
        pass

    def store_challenge(self, inputs: Mapping[str, Any], challenge_output: Challenges,
                        current_challenge: ChallengeTriplet):
        """
        Method to store challenges outputs and update the ChallengeAgentState
        :param inputs: Current state passed to the challenge agent (TypedDict, dict)
        :param challenge_output: Challenges object containing the collection of challenge information
        :param current_challenge: The current challenge object (i.e. the current question being asked)
        :return: None
        """
        updated_state = {
            "messages": [],
            "current_narrative_context": inputs["current_narrative_segment"],
            "narrative_beat_info": {"characters": [], "theme": "Fun", "tone": "happy",
                                    "plot": "Starting the adventure"},
            "story_history": inputs["story_history"] + [f"Challenge Master: {challenge_output}"],
            "challenge_type": "Vocabulary Awareness",
            "modality": "Text/Audio",
            "challenge_history": [current_challenge]
        }

        # update the ChallengeAgentState with the new information
        for k, v in updated_state.items():
            self.update_state(k, v)

        # Todo add logic to store challenges in a challenge database or in memory

    def get_previous_challenge(self):
        """
        Placeholder to get previous challenges from the memory or the current state.
        :return:
        """
        pass

    def update_modality_constraints(self, updated_constraints: dict):
        """
        Updates the modality constraint object containing information about how challenges should be presented
        :param updated_constraints: A dictionary containing the updated information
        :return: None
        """
        self.modality_constraint = {**self.modality_constraint, **updated_constraints}

    def update_narrative_constraints(self, updated_constraints: dict):
        """
        Updates the narrative constraint object containing limiting information so that challenges make sense in the
        context of the narrative.
        :param updated_constraints: A dictionary containing the updated information
        :return: None
        """
        self.narrative_constraints = {**self.narrative_constraints, **updated_constraints}