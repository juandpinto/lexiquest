
from langchain_ollama import ChatOllama

from .utils import BaseAgent
from core.states import FullState, AssessmentState
from core.assessments import ChallengePairings, ResponseEvaluation





SUBTASK1_DESCRIPTION = """
Each challenge presents a triplet of words (e.g. "dog-cat-bone") and the student is asked to
choose 2 words that go together and justify their choice. For each triplet, the student must
provide 2 pairs with justifications for each.

Examples:
dog-cat-bone: (dog, cat), because they are both animals
              (dog, bone), because dogs like bones

light-sun-feather: (light, sun), because the sun produces light
                   (light, feather), because a feather is light / not heavy
"""



SUBTASK1_EXTRACTION_PROMPT = """
You are given student responses to Vocabulary Awareness (VA) challenges.

Your task is to extract the pairs and their justifications from the given student response.

Example:
- Student Response: "I think its dog and cat because they are animals and dog and bone because dogs like bones."
- Output:  (dog, cat): "they are animals"
           (dog, bone): "dogs like bones"

"""



SUBTASK1_EVALUATION_PROMPT = """
You are responsible for evaluating student responses to Vocabulary Awareness (VA) challenges.

{task_description}

Your task is to verify whether the student's selected word pair and their justification for each pair is valid, given the expected response.
If BOTH word pair and justification are correct, score 1, else score 0. Then, compute the total score as the sum of scores for each evaluated pair. The maximum total score is 2.

Example:
- Student Response:
    (light, sun): light comes from sun
    (light, feather): feathers are light

- Expected Response:
light-sun-feather: (light, sun): because sun gives light / both are bright
                   (light, feather): because feather is light / not heavy

- Output:
    (light, sun): True
    "light comes from sun": True
    Score: 1

    (feather, light): True
    "feathers are light": True
    Score: 1

    Total Score = 1 + 1 = 2
"""



class AssessmentAgent(BaseAgent):

    def __init__(self, model):
        super().__init__(name="Assessment Agent")

        self.model = model
        self.item_total_scores = []
        self.basal_move_backwards = False
        self.ceiling_stop_subtask = False

        self.state: AssessmentState = {
            "basal": False,
            "ceiling": False,
            "evaluated_pairings": []
        }

        self.prompt_template = """
            You are an expert educational evaluator in an interactive storytelling game.

            Subtask Description:
            {subtask_description}

            Subtask Instructions:
            {subtask_instructions}

            Input:
            {input}
        """



    def __call__(self, fullstate: FullState) -> FullState:
        """
        Main callable interface for the Assessment Agent.
        Evaluates student response to a single subtask 1 (VA) item.

        Args:
            state (FullState): The current state dictionary containing student response,
            expected answer, scoring history, and other agent context.

        Returns:
            updated_state (FullState): The updated state including evaluation results, score for the
            current item, and updated flags (e.g., basal, response mode).
        """

        print("--- Running Assessment Agent ---")


        student_response = fullstate.student_response
        expected_response = fullstate.expected_response

        # todo: handle exception case if these don't exist
        extracted_pairings = self.extract_pairings(student_response)
        evaluated_pairings = self.evaluate_pairings(extracted_pairings, expected_response)

        # basal rule
        if len(self.item_total_scores) < 4:
            self.basal_move_backwards = self.check_basal_rule()

        # ceiling rule
        if len(self.item_total_scores) >= 8:
            self.ceiling_stop_subtask = self.check_ceiling_rule()

        self.store_assessment(evaluated_pairings)
        fullstate.assessment = self.state

        return fullstate



    def extract_pairings(self, normalized_student_response: str) -> ChallengePairings:
        """
        Extracts the word pair and justification pairings from the student's response.

        Args:
            normalized_student_response (str): Speech-to-text transcription of the student's response to subtask 1 (VA)

        Returns:
            extracted_responses (ChallengePairings): List of pairings (word pair and justification)

        """

        extraction_prompt_str = self.prompt_template.format(
            subtask_description = SUBTASK1_DESCRIPTION,
            subtask_instructions = SUBTASK1_EXTRACTION_PROMPT,
            input = normalized_student_response
        )

        extraction_structured_llm = self.model.with_structured_output(ChallengePairings)
        extracted_responses = extraction_structured_llm.invoke(extraction_prompt_str)

        return extracted_responses



    def evaluate_pairings(self, student_response: ChallengePairings, expected_response: ChallengePairings) -> ResponseEvaluation:
        """
        Evaluates student's response to the given subtask 1 (VA) challenge.
        Computes the score for each item in subtask 1. Score 1 for each correct word pair and reason (both must be correct), else score 0.

        Args:
            student_response (ChallengePairings): List of student's selected pairings (word pair and justification)
            expected_response (ChallengePairings): List of expected pairings (word pair and justification)

        Returns:
            evaluated_responses (ResponseEvaluation): List of evaluations for each pairing
        """

        response_block = f"Student Response: {student_response}\n\nExpected Response: {expected_response}"

        eval_prompt_str = self.prompt_template.format(
            subtask_description = SUBTASK1_DESCRIPTION,
            subtask_instructions = SUBTASK1_EVALUATION_PROMPT,
            input = response_block
        )

        eval_structured_llm = self.model.with_structured_output(ResponseEvaluation)
        evaluated_responses = eval_structured_llm.invoke(eval_prompt_str)

        evaluated_responses.update_total_score()
        self.item_total_scores.append( int(evaluated_responses.total_score) )

        return evaluated_responses



    def check_basal_rule(self):
        """
        Determines whether the starting point of the subtask needs to be moved backwards.
        TILLS Basal Rule for subtask 1: Four consecutive scores of 2 (both parts of each item must be correct)

        Returns:
            True if starting point needs to be moved back, otherwise False
        """

        return self.item_total_scores[-1] < 2



    def check_ceiling_rule(self):
        """
        Determines the stopping point of the subtask.
        TILLS Ceiling Rule for subtask 1: Six scores of 0 within a sequence of eight consecutive items
                            (both parts of each item must be incorrect on 6 out of 8 items)

        Returns:
            True if stopping point has been reached, otherwise False
        """

        return self.item_total_scores[-8:].count(0) >= 6



    def store_assessment(self, evaluated_pairings: ResponseEvaluation):
        """
        Stores the assessment results of the current item under subtask 1.

        Args:
            evaluated_pairings (ResponseEvaluation): List of evaluations for each pairing
        """

        self.state.update({
            "basal": self.basal_move_backwards,
            "ceiling": self.ceiling_stop_subtask,
            "evaluated_pairings": self.state["evaluated_pairings"] + [evaluated_pairings]
        })

        # Todo: store in memory




    def analyze_behavioral_indicators(self):
        """
        Method to analyze student's response for SLD behavioral indicators.
        """
        pass



    def determine_flags(self):
        """
        Method to determine flags like retry, scaffold, fallback, etc.
        """
        pass






def pretty_print_assessment_state(assessment_state):

    print("\n🧠 Assessment Summary\n" + "=" * 30)

    for i, response in enumerate(assessment_state['evaluated_pairings']):
        print(f"\n📘 Item {i + 1}")
        print("-" * 20)
        for j, eval in enumerate(response.evaluations):
            pair = eval.evaluated_pairing.words
            justification = eval.evaluated_pairing.justification
            print(f"  Pair {j + 1}: {pair}")
            print(f"    Justification: \"{justification}\"")
            print(f"    Pair Valid: {eval.pair_is_valid}")
            print(f"    Justification Valid: {eval.justification_is_valid}")
            print(f"    Score: {eval.score.value}")
        print(f"  Total Score: {response.total_score.value}")



def test_assessment_agent():

    llm = ChatOllama(model="gemma3", temperature=0.8)
    agent = AssessmentAgent(model=llm)

    # Sample student and expected responses
    student_response = (
        "I think it's pen and paper because you use a pen to write on paper, "
        "and pen and pig because you keep a pig in a pen."
    )

    expected_response = {
        "pairings": [
            {"words": ["pen", "paper"], "justification": "you use a pen to write on paper"},
            {"words": ["pen", "pig"], "justification": "you keep a pig in a pen"}
        ]
    }


    state = FullState()
    state.student_response = student_response
    state.expected_response = expected_response

    new_state = agent(state)

    pretty_print_assessment_state(new_state.assessment)



if __name__ == "__main__":
    test_assessment_agent()
