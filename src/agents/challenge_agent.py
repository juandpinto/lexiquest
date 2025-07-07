import pprint
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Any, Mapping
from abc import ABC, abstractmethod
from core.states import ChallengeState, FullState
from core.challenges import BaseChallenge
from langchain_core.messages import AIMessage

SUBTASK1_INSTRUCTION_PROMPT = """
Subtest 1 is concerned with evaluating a students Vocabulary Awareness (VA). The test consists of presenting a student with a set of
3 words (e.g. dog-cat-bone) and asking them "Tell me two words that go together" and then ask for their justification. For each set, students must come up
with 2 pairings along with their associated justifications. Students are allowed repeated attempts at any given item.

Examples:
dog-cat-bone: (dog, cat), because they are both animals
              (dog, bone), because dogs like bones

light-sun-feather: (light, sun), because the sun produces light
                   (light, feather), because a feather is light / not heavy
"""
SUBTASK1_INSTRUCTION_CONSTRAINTS = """
- Each challenge presents a word triplet (x-y-z) and asks the student to choose words that go together and explain why.
- For each triplet, generate exactly 2 correct pairings with associated justifications.
- Justifications should be based on the actual meaning of words and nothing else
- Triplets should be chosen based on their contextual relationship with the narrative
- Do not define words.
- Try to avoid ambiguity, at least one pair within any triplet should explicitly not match.
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
        {challenge_schema}
        ```
        """

        self.state: ChallengeState = ChallengeState(
            messages=[],
            current_narrative_segment=[],
            narrative_beat_info={},
            challenge_type="",
            modality="",
            story_history=[],
            challenge_history=[]
        )
        self.narrative_constraints = NarrativeConstraint
        self.modality_constraint = {}
        self.current_challenge = 0

    def __call__(self, inputs: FullState) -> FullState:
        print("[challenge_agent] Input state:", inputs)
        if len(inputs.full_history) == 0:
            # Store the error in the state instead of returning a dict
            inputs.full_history.append(AIMessage(content="Error: No narrative to base challenge on."))
            print("[challenge_agent] Output state:", inputs)
            print("[challenge_agent] Output type:", type(inputs))
            print("\n--- Exiting Challenge Agent ---")
            return inputs
        challenge_output = self.generate_challenge(inputs)
        current_challenge = challenge_output[self.current_challenge]
        print("--- Completed Challenge Query ---")
        print(current_challenge)
        print()

        self.store_challenge(inputs, challenge_output, current_challenge)
        print("[challenge_agent] Output state:", inputs)
        print("[challenge_agent] Output type:", type(inputs))
        print("\n--- Exiting Challenge Agent ---")
        return inputs

    def generate_challenge(self, inputs: FullState) -> list:
        """
        Generates a Challenges object which will contain question information such as the question, the answer, and the
        justification for the answer if necessary.
        :param inputs: Current state passed to the challenge agent (FullState object)
        :return list: A challenge plan with a list of challenges
        """
        context_input = {
            "current_narrative_context": inputs.narrative.story[-1].content,
            "story_history": str(inputs.full_history[-1].content).replace('{', '{{').replace('}', '}}').replace("'", '"'),
            "subtask_instruction": SUBTASK1_INSTRUCTION_PROMPT,
            "subtask_constraints": SUBTASK1_INSTRUCTION_CONSTRAINTS,
            "challenge_type": "Vocabulary Awareness",
            "modality": "Text/Audio",
        }
        # Todo setup better logic for switching between challenge types
        current_challenge_schema = BaseChallenge.get_example_for('Vocabulary Awareness')
        structured_output_parser = BaseChallenge.get_class_by_type('Vocabulary Awareness')
        current_challenge_schema_str = str(pprint.pformat(current_challenge_schema))\
            .replace('{', '{{').replace('}', '}}').replace("'", '"')
        self.output_schema = self.output_schema.format(challenge_schema=current_challenge_schema_str).strip()

        # Format the current challenge
        challenge_prompt = self.challenge_prompt_template.format(**context_input).strip() + "\n" + self.output_schema

        prompt = ChatPromptTemplate([
            ("system", challenge_prompt),
            # ("system", "Previous challenges:\n{chat_history}"),
            ("human", "{query}")
        ])

        # Force the model to provide structure output for ease of use and consistency
        model = self.model.with_structured_output(structured_output_parser.example().class_type())

        # Todo not sure if this is the best initial query-system_prompt combination
        query = "Generate a challenge based on the current current narrative context and subtask." \
                "Do not create duplicate challenges if previous challenges are given."

        print("--- Starting Challenge Query ---")
        print(f'--- Input Prompt: {prompt} ---')

        # Create query chain to get output from the model
        chain = prompt | model
        challenge_history = []
        for i in range(5):
            challenge = chain.invoke({"query": query})
            challenge_history.append(challenge)

            prev_challenges = "\n".join([
                str(x).replace('{', '{{').replace('}', '}}').replace("'", '"')
                for x in challenge_history
            ])
            query = query + "\n\nPrevious Challenges:" if i == 0 else "\n\n" + prev_challenges

        return challenge_history  # Assuming nothing went wrong should return a list of BaseChallenge object

    def validate_challenge(self):
        """
        Placeholder function to validate whether questions within a challenge make sense in the context of the narrative
        and subtask
        :return:
        """
        pass

    def store_challenge(self, inputs: FullState, challenge_output: list,
                        current_challenge: BaseChallenge):
        """
        Method to store challenges outputs and update the ChallengeAgentState
        :param inputs: Current state passed to the challenge agent (TypedDict, dict)
        :param challenge_output: Challenges object containing the collection of challenge information
        :param current_challenge: The current challenge object (i.e. the current question being asked)
        :return: None
        """
        updated_state = {
            "messages": [],
            "current_narrative_segment": inputs.narrative.story,
            "narrative_beat_info": {"characters": [], "theme": "Fun", "tone": "happy",
                                    "plot": "Starting the adventure"},
            "story_history": inputs.full_history + [AIMessage(content=f"Challenge Master: {current_challenge}")],
            "challenge_type": "Vocabulary Awareness",
            "modality": "Text/Audio",
            "challenge_history":  self.state.challenge_history + challenge_output
        }

        # update the ChallengeAgentState with the new information
        for k, v in updated_state.items():
            setattr(self.state, k, v)
        setattr(inputs, "challenge", self.state)
        setattr(inputs, "full_history",
                inputs.full_history + [AIMessage(content=f"Challenge Master: {challenge_output}")])

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
