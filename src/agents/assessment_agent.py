
from langchain_ollama import ChatOllama

from utils import BaseAgent
from prompts import ASSESSMENT_PROMPTS
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
        """
        
    

    def __call__(self, challenge_index: int, fullstate: FullState) -> FullState:
        """
        Main callable interface for the Assessment Agent.
        Evaluates student response to a TILLS subtask challenge.

        Args:
            challenge_index (int): Index of the challene being evaluated.
            state (FullState): The current state dictionary containing student response, expected answer, scoring history, 
                                and other agent context.

        Returns:
            updated_state (FullState): The updated state including evaluation results, score for the 
            current item, and updated flags (e.g., basal, response mode).
        """

        subtask_key = fullstate.challenge.challenge_type # e.g Vocabulary Awareness
        subtask_handler = self.get_subtask(subtask_key)
        
        raw_student_response = fullstate.student_response
        challange_item = fullstate.challenge.challenge_history[challenge_index]

        print("--- Running Assessment Agent ---")

        # todo: handle exception case if these don't exist
        extracted_student_answer = self.extract_student_answers(subtask_handler, raw_student_response)
        evaluated_student_answer = self.evaluate_student_answers(subtask_handler, extracted_student_answer, challange_item) 

        self.basal_move_backwards = self.check_basal_rule(subtask_handler)
        self.ceiling_stop_subtask = self.check_ceiling_rule(subtask_handler)
        
        self.store_assessment(evaluated_student_answer)
        fullstate.assessment = self.state

        return fullstate
    



    def get_subtask(self, subtask_key: str) -> BaseAssessmentSubtask:
        """
        Retrives a handler for the current TILLS subtask.

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
            extracted_student_answers (BaseAssessmentExtractSchema): Student's answers to the current challenge
        
        """

        formatted_input = subtask_handler.format_extraction_input(raw_student_response)

        extraction_prompt_str = self.prompt_template.format(
            subtask_description = ASSESSMENT_PROMPTS[subtask_handler.type_key]["description"],
            subtask_instructions = ASSESSMENT_PROMPTS[subtask_handler.type_key]["extraction"],
            input = formatted_input
        )

        extraction_structured_llm = self.model.with_structured_output(subtask_handler.extraction_schema)
        extracted_student_answer = extraction_structured_llm.invoke(extraction_prompt_str)

        return extracted_student_answer



    def evaluate_student_answers(self, subtask_handler: BaseAssessmentSubtask, extracted_student_answers: BaseAssessmentExtractSchema, challange_item: BaseChallenge) -> BaseAssessmentEvalSchema:
        """
        Evaluates student's answers to the given subtask challenge.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.
            extracted_student_answers (BaseAssessmentExtractSchema): Student's answers to the current challenge
            challenge_item (BaseChallenge): The ground-truth challenge data for the current subtask item

        Returns:
            evaluated_student_answers (BaseAssessmentEvalSchema): List of evaluations for each pairing
        """

        formatted_input = subtask_handler.format_evaluation_input(extracted_student_answers, challange_item)

        eval_prompt_str = self.prompt_template.format(
            subtask_description = ASSESSMENT_PROMPTS[subtask_handler.type_key]["description"],
            subtask_instructions = ASSESSMENT_PROMPTS[subtask_handler.type_key]["evaluation"],
            input = formatted_input
        )

        eval_structured_llm = self.model.with_structured_output(subtask_handler.evaluation_schema)
        evaluated_student_answers = eval_structured_llm.invoke(eval_prompt_str)
        
        eval_ans_total_score = subtask_handler.update_score(evaluated_student_answers)
        self.item_total_scores.append(eval_ans_total_score) 

        return evaluated_student_answers

    

    def check_basal_rule(self, subtask_handler):
        """
        Determines whether the starting point of the subtask needs to be moved backwards, 
        based on the subtask-specific basal rule.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask. 
    
        Returns:
            True if starting point needs to be moved back, otherwise False
        """

        return subtask_handler.check_basal_rule(self.item_total_scores)
        


    def check_ceiling_rule(self, subtask_handler):
        """
        Determines the stopping point of the subtask, based on the subtask-specific ceiling rule.

        Args:
            subtask_handler (BaseAssessmentSubtask): The handler for the current subtask.

        Returns:
            True if stopping point has been reached, otherwise False
        """

        return subtask_handler.check_ceiling_rule(self.item_total_scores)
        


    def store_assessment(self, evaluated_student_answers: BaseAssessmentEvalSchema):
        """
        Stores the assessment results of the current subtask challenge item.

        Args:
            evaluated_student_answers (BaseAssessmentEvalSchema): List of challenge item evaluations.
        """

        self.state.update({
            "basal": self.basal_move_backwards,
            "ceiling": self.ceiling_stop_subtask,
            "assessment_history": self.state["assessment_history"] + [evaluated_student_answers]
        })

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
            "assessment_history": []            
        }


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


    def give_feedback(self):
        """
        Method to send feedback back to the manager agent.
        """
        pass


    





# Todo: generalze test code to all subtasks

def pretty_print_assessment_state(assessment_state):

    print("\n🧠 Assessment Summary\n" + "=" * 30)

    for i, response in enumerate(assessment_state['assessment_history']):
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

    challenge_item = ChallengeTriplet(
        triplet=("pen", "paper", "pig"),
        pairings=[
            Pairing(words=("pen", "paper"), justification="you use a pen to write on paper"),
            Pairing(words=("pen", "pig"), justification="you keep a pig in a pen")
        ]
    )

    full_state = FullState()
    full_state.challenge.challenge_type = "Vocabulary Awareness"
    full_state.challenge.challenge_history.append(challenge_item)

    full_state.student_response = {
        "alphabetic": (
            "I think it's pen and paper because to write on paper we need a pen, "
            "and pen and pig because you keep a pig inside a pig pen."
        )
    }

    updated_state = agent(challenge_index=0, fullstate=full_state)

    pretty_print_assessment_state({
        "assessment_history": updated_state.assessment["assessment_history"]
    })



if __name__ == "__main__":
    test_assessment_agent()