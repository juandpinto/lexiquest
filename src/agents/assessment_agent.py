import os
import json

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel
from .utils import BaseAgent
from .prompts import ASSESSMENT_PROMPTS
from core.states import FullState, AssessmentState
from core.challenges import BaseChallenge, Pairing, ChallengeTriplet
from core.assessments import BaseAssessmentSubtask, BaseAssessmentExtractSchema, BaseAssessmentEvalSchema




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
            "score_summary": {},
            "assessment_history": []
        }

        self.prompt_template = """
            You are an expert educational evaluator in an interactive storytelling game.

            Subtask Description:
            {subtask_description}

            Subtask Instructions:
            {subtask_instructions}

            Input:
            {input}

            Use the following **JSON schema** for output (do not include anything other than the JSON):

            ```json
            {schema}
            ```
        """


    
    def __call__(self, fullstate: FullState) -> FullState:
        """
        Main callable interface for the Assessment Agent.
        Evaluates student response to a subtask challenge.

        Args:
            state (FullState): The current state dictionary containing student response, expected answer, scoring history,
                                and other agent context.

        Returns:
            fullstate (FullState): The updated state including evaluation results, score for the
                                    current item, and updated flags (e.g., basal, response mode).
        """

        challenge_index = fullstate.narrative.challenge_index

        subtask_key = fullstate.challenge.challenge_type 
        subtask_handler = self.get_subtask(subtask_key)

        raw_student_response = fullstate.student_response
        challenge_item = fullstate.challenge.challenge_history[challenge_index]

        print("\n--- Running Assessment Agent ---")

        # todo: handle exception case if these don't exist
        extracted_student_answer = self.extract_student_answers(subtask_handler, raw_student_response)
        evaluated_student_answer = self.evaluate_student_answers(subtask_handler, extracted_student_answer, challenge_item)

        self.basal_move_backwards = self.check_basal_rule(subtask_handler)
        self.ceiling_stop_subtask = self.check_ceiling_rule(subtask_handler)

        self.store_assessment(subtask_handler, evaluated_student_answer)
        fullstate.assessment = self.state

        fullstate.assessment_feedback = self.generate_feedback()

        return fullstate




    def get_subtask(self, subtask_key: str) -> BaseAssessmentSubtask:
        """
        Retrives a handler for the current subtask.

        Args:
            subtask_key (str): Unique identifier of the subtask

        Returns:
            BaseAssessmentSubtask: An initialized handler for the subtask
        """

        subtask_class = BaseAssessmentSubtask.get_cls_by_key(subtask_key)
        return subtask_class()




    def extract_student_answers(self, subtask_handler: BaseAssessmentSubtask, raw_student_response: str) -> BaseAssessmentExtractSchema:
        """
        Extracts the subtask answers from the student's raw response.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.
            raw_student_response (str): Transcription of the student's raw response to the given challenge.

        Returns:
            extracted_student_answer (BaseAssessmentExtractSchema): Student's answers to the current challenge

        """

        formatted_input = subtask_handler.format_extraction_input(raw_student_response)

        extraction_prompt_str = self.prompt_template.format(
            subtask_description = ASSESSMENT_PROMPTS[subtask_handler.type_key]["description"],
            subtask_instructions = ASSESSMENT_PROMPTS[subtask_handler.type_key]["extraction"],
            input = formatted_input,
            schema = self.get_schema_block(subtask_handler, "extraction")
        )

        extraction_structured_llm = self.model.with_structured_output(subtask_handler.extraction_schema)
        extracted_student_answer = extraction_structured_llm.invoke(extraction_prompt_str)

        extracted_student_answer = subtask_handler.filter_extracted_answers(extracted_student_answer, raw_student_response)

        return extracted_student_answer



    def evaluate_student_answers(self, subtask_handler: BaseAssessmentSubtask, extracted_student_answers: BaseAssessmentExtractSchema, challenge_item: BaseChallenge) -> BaseAssessmentEvalSchema:
        """
        Evaluates student's answers to the given subtask challenge.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.
            extracted_student_answers (BaseAssessmentExtractSchema): Student's answers to the current challenge
            challenge_item (BaseChallenge): The ground-truth challenge data for the current subtask item

        Returns:
            evaluated_student_answers (BaseAssessmentEvalSchema): List of evaluations for each challenge item
        """

        formatted_input = subtask_handler.format_evaluation_input(extracted_student_answers, challenge_item)

        eval_prompt_str = self.prompt_template.format(
            subtask_description = ASSESSMENT_PROMPTS[subtask_handler.type_key]["description"],
            subtask_instructions = ASSESSMENT_PROMPTS[subtask_handler.type_key]["evaluation"],
            input = formatted_input,
            schema = self.get_schema_block(subtask_handler, "extraction")
        )

        eval_structured_llm = self.model.with_structured_output(subtask_handler.evaluation_schema)
        evaluated_student_answers = eval_structured_llm.invoke(eval_prompt_str)

        eval_ans_total_score = subtask_handler.update_score(evaluated_student_answers)
        self.item_total_scores.append(eval_ans_total_score)

        return evaluated_student_answers

    

    def check_basal_rule(self, subtask_handler: BaseAssessmentSubtask):
        """
        Determines whether the starting point of the subtask needs to be moved backwards,
        based on the subtask-specific basal rule.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.

        Returns:
            True if starting point needs to be moved back, otherwise False
        """

        return subtask_handler.check_basal_rule(self.item_total_scores)



    def check_ceiling_rule(self, subtask_handler: BaseAssessmentSubtask):
        """
        Determines the stopping point of the subtask, based on the subtask-specific ceiling rule.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.

        Returns:
            True if stopping point has been reached, otherwise False
        """

        return subtask_handler.check_ceiling_rule(self.item_total_scores)



    def store_assessment(self, subtask_handler: BaseAssessmentSubtask, evaluated_student_answers: BaseAssessmentEvalSchema):
        """
        Stores the assessment results of the current subtask challenge item.

        Args:
            evaluated_student_answers (BaseAssessmentEvalSchema): List of challenge item evaluations.
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.
        """

        if subtask_handler.type_key not in self.state["score_summary"]:
            self.state["score_summary"] = {
                "total_items": 0,
                "total_score": 0,
                "average_score": 0.0
            }

        summary = self.state["score_summary"]

        summary["total_items"] += 1
        summary["total_score"] += self.item_total_scores[-1]
        summary["normalized_average"] = round(
            summary["total_score"] / (summary["total_items"] * subtask_handler.max_item_score), 2
        )

        self.state.update({
            "basal": self.basal_move_backwards,
            "ceiling": self.ceiling_stop_subtask,
            "score_summary": summary,
            "assessment_history": self.state["assessment_history"] + [evaluated_student_answers]
        })


        # TODO: save for each user, prevent file overwrite
        subtask_handler.export_to_csv_and_plots(self.state["assessment_history"], self.state["score_summary"])

       
        # Todo: store in memory



    def reset(self):
        """
        Method to reset agent state.
        """

        self.subtask = ""
        self.item_total_scores = []
        self.basal_move_backwards = False
        self.ceiling_stop_subtask = False

        self.state: AssessmentState = {
            "basal": False,
            "ceiling": False,
            "score_summary": {},
            "assessment_history": []
        }



    def generate_feedback(self):
        """
        Sends feedback back to the manager agent.
        """

        if self.basal_move_backwards and self.ceiling_stop_subtask:
            return "Student showed signs of struggle with inital items and also reached the ceiling criterion."

        elif self.basal_move_backwards:
            return "Student did not meet the basal criterion and struggled with initial items."

        elif self.ceiling_stop_subtask:
            return "Student reached the ceiling criterion; subtask was appropriately concluded."

        else:
            return "Student completed the subtask successfully without triggering basal or ceiling limits."



    def get_schema_block(self, subtask_handler: BaseAssessmentSubtask, task: str):
        """
        Retrieves the required Pydantic schema and converts to json.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.

        Returns:
            schema (json): The Pydantic schema in json format.
        """

        if task == "extraction":
            schema = subtask_handler.extraction_schema.model_json_schema()

        elif task == "evaluation":
            schema = subtask_handler.evaluation_schema.model_json_schema()

        return json.dumps(schema, separators=(",", ":"))



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










# Todo: generalze test code to all subtasks

def pretty_print_assessment_state(assessment_state):

    print("\n\U0001F9E0 Assessment Summary\n" + "=" * 30)

    for i, response in enumerate(assessment_state['assessment_history']):
        print(f"\n\U0001F4D8 Item {i + 1}")
        print("-" * 20)
        for j, eval in enumerate(response.evaluations):
            pair = eval.evaluated_pairing.words
            justification = eval.evaluated_pairing.justification
            error = eval.error_analysis.category
            error_reason = eval.error_analysis.category_reasoning
            print(f"  Pair {j + 1}: {pair}")
            print(f"    Justification: \"{justification}\"")
            print(f"    Pair Valid: {eval.pair_is_valid}")
            print(f"    Justification Valid: {eval.justification_is_valid}")
            print(f"    Score: {eval.score.value}")
            print(f"    Error Category: {error.value}")
            print(f"    Error Category Reasoning: {error_reason}")
        print(f"  Total Score: {response.total_score.value}")

def test_assessment_agent():
    llm = ChatOllama(model="gemma3", temperature=0.8)
    # api_key = os.getenv("GOOGLE_API_KEY")
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.8)
    agent = AssessmentAgent(model=llm)

    challenges = [
        ChallengeTriplet(
            triplet=("pen", "paper", "pig"),
            pairings=[
                Pairing(words=("pen", "paper"), justification="you use a pen to write on paper"),
                Pairing(words=("pen", "pig"), justification="you keep a pig in a pen")
            ]
        ),
        ChallengeTriplet(
            triplet=("sun", "moon", "light"),
            pairings=[
                Pairing(words=("sun", "light"), justification="the sun gives off light"),
                Pairing(words=("moon", "sun"), justification="both are in the sky")
            ]
        ),
        ChallengeTriplet(
            triplet=("dog", "cat", "bone"),
            pairings=[
                Pairing(words=("dog", "cat"), justification="both are common pets"),
                Pairing(words=("dog", "bone"), justification="dogs like bones")
            ]
        ),

    ]

    responses = [
        "pen and paper because you write with a pen, and pen and pig because pigs live in pens.",
        "sun and light because the sun makes light, and sun and moon because they're both in the sky.",
        "cat and bone because cats chew bones, and cat and dog because they are animals.",
    ]

    full_state = FullState()
    full_state.challenge.challenge_type = "Vocabulary Awareness"

    for challenge, response in zip(challenges, responses):
        full_state.challenge.challenge_history.append(challenge)
        full_state.student_response = response
        full_state.narrative.challenge_index = len(full_state.challenge.challenge_history) - 1
        full_state = agent(fullstate=full_state)

    pretty_print_assessment_state({
        "assessment_history": full_state.assessment["assessment_history"]
    })

    print("\n\nFeedback Summary:")
    print(full_state.assessment_feedback)

    print('\n\nFull Assessment State Summary:\n')
    print(full_state.assessment)


if __name__ == "__main__":
    test_assessment_agent()

