from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Optional, ClassVar, Dict, Type

from pydantic import BaseModel, Field

from core.challenges import Pairing, BaseChallenge




class ItemScoreEnum(int, Enum):
    incorrect = 0
    correct = 1



class BaseAssessmentExtractSchema(BaseModel):
    """The Pydantic schema for a TILLS subtask's extracted response"""
    pass



class BaseAssessmentEvalSchema(BaseModel):
    """The Pydantic schema for a TILLS subtask's evaluated response"""
    pass



class BaseAssessmentSubtask(ABC):
    _registry: ClassVar[Dict[str, Type["BaseAssessmentSubtask"]]] = {}
    
    type_key: str
    extraction_schema: Type["BaseAssessmentExtractSchema"]
    evaluation_schema: Type["BaseAssessmentEvalSchema"]

    def __init_subclass__(cls):
        if hasattr(cls, "type_key"):
            BaseAssessmentSubtask._registry[cls.type_key] = cls


    @classmethod
    def get_cls_by_key(cls, key: str) -> Type["BaseAssessmentSubtask"]:
        return cls._registry[key]
    

    @abstractmethod
    def format_extraction_input(self, raw_student_response: Dict[str, any]) -> str:
        """
        Formats the student's raw response for the extraction prompt.

        Args:
            raw_student_response (Dict): Student's raw response to the given subtask challenge item

        Returns:
            formatted_extraction_input (str): Student response formatted for the extraction prompt
        """

        raise NotImplementedError
    

    @abstractmethod
    def format_evaluation_input(self, structured_student_response: Type["BaseAssessmentExtractSchema"], challenge_item: BaseChallenge) -> str:
        """
        Formats the extracted student answers and correct answers for evaluation.

        Args:
            structured_student_response (BaseAssessmentExtractSchema): Student answers to the given subtask challenge item
            challenge_item (BaseChallenge): The ground-truth challenge data for the current subtask item

        Returns:
            formatted_eval_input (str): Student and correct answers formatted for the evaluation prompt
        """
        
        raise NotImplementedError
    

    @abstractmethod
    def update_score(self, evaluated_student_answers: Type["BaseAssessmentEvalSchema"]) -> int:
        """
        Computes the total score for a subtask challenge item.

        Args:
            evaluated_student_answers (BaseAssessmentEvalSchema): Evaluated student answers for the given subtask challenge.

        Returns:
            item_total_score (int): Student's total score for the current subtask challenge item.
        """
        
        raise NotImplementedError
    

    @abstractmethod
    def check_basal_rule(self, scores: list[int]) -> bool:
        """
        Checks the subtask's basal rule.
        
        Args:
            scores (List): List of the student's scores for each challenge item under the current subtask.

        Returns:
            move_back (bool): True if starting point needs to be moved back, otherwise False
        """
        
        raise NotImplementedError
    
    
    @abstractmethod
    def check_ceiling_rule(self, scores: list[int]) -> bool:
        """
        Checks the subtask's ceiling rule.
        
        Args:
            scores (List): List of the student's scores for each challenge item under the current subtask.

        Returns:
            stop_challenge (bool): True if stopping point has been reached, otherwise False
        """
        
        raise NotImplementedError






############ SUBTASK 1: Vocabulary Awareness ############


class VAItemScoreEnum(int, Enum):
    zero = 0
    one = 1
    two = 2



class VAPairingList(BaseAssessmentExtractSchema):
    pairings: List[Pairing] = Field(
        description="List of word pairs with their associated justification for a specific triplet."
    )




class VAPairingEvaluation(BaseModel):
    evaluated_pairing: Pairing = Field(
        description="The student's selected word pair along with the justification."
    )
    
    pair_is_valid: bool = Field(
        description="True if the selected pair reflects a valid semantic relationship."
    )
    
    justification_is_valid: bool = Field(
        description="True if the justification meaningfully and clearly supports the selected pair."
    )
    
    score: ItemScoreEnum = Field(
        description="Numerical score for this response: 1 = correct, if both word pair AND justification are correct; 0 = incorrect, if word pair and/or justification is incorrect."
    )




class VAItemEvaluation(BaseAssessmentEvalSchema):
    evaluations: List[VAPairingEvaluation] = Field(
        description="List of evaluations for each word pair and their associated justification."
    )

    total_score: Optional[VAItemScoreEnum] = Field(
        default=None,
        description="Total score for this item, computed as the sum of scores for each evaluated pair. Range: 0–2."
    )
    
    def update_total_score(self):

        assert len(self.evaluations) == 2, f"Expected 2 pairings, got {len(self.evaluations)}.\n"
        
        total = sum(eval.score.value for eval in self.evaluations)
        self.total_score = VAItemScoreEnum(total)





class VocabularyAwarenessSubtask(BaseAssessmentSubtask):
    type_key = "Vocabulary Awareness"
    extraction_schema = VAPairingList
    evaluation_schema = VAItemEvaluation


    def format_extraction_input(self, raw_student_response):
        return raw_student_response["alphabetic"]



    def format_evaluation_input(self, structured_student_response, challenge_item):

        triplet_line = str(challenge_item.triplet)

        student_lines = "\n".join(
            f"({pair.words[0]}, {pair.words[1]}): {pair.justification}"
            for pair in structured_student_response.pairings
        )

        expected_lines = "\n".join(
            f"({pair.words[0]}, {pair.words[1]}): {pair.justification}"
            for pair in challenge_item.pairings
        )

        return (
            f"Triplet: {triplet_line}\n\n"
            f"Student Response: {student_lines}\n\n"
            f"Expected Response: {expected_lines}"
        )
        
    

    def update_score(self, evaluated_student_answers):

        evaluated_student_answers.update_total_score()
        return int(evaluated_student_answers.total_score)



    def check_basal_rule(self, scores):
        """
        TILLS Basal Rule for subtask 1: Four consecutive scores of 2 (both parts of each item must be correct)
        """

        return scores[-1] < 2 if len(scores) < 4 else False
        


    def check_ceiling_rule(self, scores):
        """
        TILLS Ceiling Rule for subtask 1: Six scores of 0 within a sequence of eight consecutive items 
                            (both parts of each item must be incorrect on 6 out of 8 items)
        """

        return scores[-8:].count(0) >= 6 if len(scores) >= 8 else False
    