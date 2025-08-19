import os
import csv
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, ClassVar, Dict, Type

from pydantic import BaseModel, Field

from core.challenges import ChallengeTriplet, Pairing, BaseChallenge




class ItemScoreEnum(str, Enum):
    incorrect = "0"
    correct = "1"



class BaseAssessmentExtractSchema(BaseModel):
    """The Pydantic schema for a subtask's extracted response"""
    pass



class BaseAssessmentEvalSchema(BaseModel):
    """The Pydantic schema for a subtask's evaluated response"""
    pass



class BaseAssessmentErrorAnalysisSchema(BaseModel):
    """The Pydantic schema for a subtask's error pattern analysis"""
    pass


class BaseAssessmentSubtask(ABC):
    _registry: ClassVar[Dict[str, Type["BaseAssessmentSubtask"]]] = {}

    type_key: str
    extraction_schema: Type["BaseAssessmentExtractSchema"]
    evaluation_schema: Type["BaseAssessmentEvalSchema"]
    error_analysis_schema: Type["BaseAssessmentErrorAnalysisSchema"]

    max_item_score: int

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
    def filter_extracted_answers(self, structured_student_response: Type["BaseAssessmentExtractSchema"], raw_student_response: Dict[str, any]) -> Type["BaseAssessmentExtractSchema"]:
        """
        Filters the extracted student answers to ensure only correct responses are present.

        Args:
            raw_student_response (Dict): Student's raw response to the given subtask challenge item
            structured_student_response (BaseAssessmentExtractSchema): Student answers to the given subtask challenge item

        Returns:
            filtered_student_response (BaseAssessmentExtractSchema): Filtered student answers to the given subtask challenge item
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
    

    @abstractmethod
    def export_to_csv_and_plots(self, assessment_history: list[Type["BaseAssessmentEvalSchema"]], score_summary: Dict[str, any]):
        """
        Exports the current assessment history and score summary to various plots and CSV files.

        Args:
            assessment_history (list): The history of evaluated student answers
            score_summary (dict): Summary of the current challenge assessment scores
        """

        raise NotImplementedError






############ SUBTASK 1: Vocabulary Awareness ############


class VAItemScoreEnum(str, Enum):
    zero = "0"
    one = "1"
    two = "2"



class VAErrorCategoryEnum(str, Enum):
    semantic_mismatch = "semantic_mismatch"
    justification_vague = "justification_vague"
    off_topic = "off_topic"
    incomplete = "incomplete"
    other = "other"
    none = "none"



class VAPairingList(BaseAssessmentExtractSchema):
    pairings: List[Pairing] = Field(
        max_length=2,
        description="List of 1 or 2 word pairs with their associated justification for a specific triplet."
    )



class VAErrorAnalysis(BaseAssessmentErrorAnalysisSchema):
    category: VAErrorCategoryEnum = Field(
        default="None",
        description="A short tag indicating the most likely error category in the student's incorrect response."
    )

    category_reasoning: str = Field(
        default="No incorrect response to analyze.",
        description="A brief explanation justifying why this error category was chosen."
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

    error_analysis: VAErrorAnalysis = Field(
        description="Error tagging and reasoning for incorrect responses. Resorts to default values if the response is correct."
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

        print(f"[VAItemEvaluation] Evaluations: {self.evaluations}")

        # Removing assertion below for now.
        # TODO: Incorporate a way for the narrative agent to encourage the student to make 2 pairs if only one is made.
        # assert len(self.evaluations) == 2, f"Expected 2 pairings, got {len(self.evaluations)}.\n"

        total = sum(int(eval.score.value) for eval in self.evaluations)
        self.total_score = VAItemScoreEnum(str(total))





class VocabularyAwarenessSubtask(BaseAssessmentSubtask):
    type_key = "Vocabulary Awareness"
    extraction_schema = VAPairingList
    evaluation_schema = VAItemEvaluation

    max_item_score = 2


    def format_extraction_input(self, raw_student_response):
        # return raw_student_response["alphabetic"]
        return raw_student_response



    def filter_extracted_answers(self, structured_student_response, raw_student_response):

        ranked = []
        pair_dict = defaultdict(list)
        student_text = raw_student_response.lower()

        for pair in structured_student_response.pairings:
            key = tuple(sorted(pair.words))
            pair_dict[key].append(pair)


        for key, candidates in pair_dict.items():
            best = max(
                candidates,
                key=lambda p: (
                    2 if p.justification.strip().lower() in student_text else 0
                    + sum(word in student_text for word in p.justification.strip().lower().split())
                )
            )

            ranked.append(best)


        return VAPairingList(pairings=ranked[:2])

                         


    def format_evaluation_input(self, structured_student_response, challenge_item):

        # Ensure that challenge_item is of type ChallengeTriplet
        if not isinstance(challenge_item, ChallengeTriplet):
            challenge_item = ChallengeTriplet.from_dict(challenge_item)

        triplet_line = str(challenge_item.triplet)

        student_lines = "\n".join(
            f"({pair.words[0]}, {pair.words[1]}): {pair.justification}"
            for pair in structured_student_response.pairings
        )

        expected_lines = "\n".join(
            f"({pair.words[0]}, {pair.words[1]}): {pair.justification}"
            for pair in challenge_item.pairings
        )

        return None if student_lines == "" else (
            f"Triplet: {triplet_line}\n\n"
            f"Student Response: {student_lines}\n\n"
            f"Expected Response: {expected_lines}"
        )



    def update_score(self, evaluated_student_answers):

        evaluated_student_answers.update_total_score()
        return int(evaluated_student_answers.total_score)



    def check_basal_rule(self, scores):
        return scores[-1] < 2 if len(scores) < 4 else False



    def check_ceiling_rule(self, scores):
        return scores[-8:].count(0) >= 6 if len(scores) >= 8 else False
    


    def export_to_csv_and_plots(self, assessment_history, score_summary, dir="src/agents/outputs/"):

        os.makedirs(dir, exist_ok=True)

        history_filename = dir + "VA_assessment_history.csv"

        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(history_filename, 'w+', newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(
                [
                    "Item", "Pair", "Justification", "Pair Valid", "Justification Valid", "Score", "Error Category", "Error Reasoning"
                ]
            )

            for i, response in enumerate(assessment_history):
                for eval in response.evaluations:
                    writer.writerow(
                        [
                            i + 1,
                            " & ".join(eval.evaluated_pairing.words),
                            eval.evaluated_pairing.justification,
                            eval.pair_is_valid,
                            eval.justification_is_valid,
                            eval.score.value,
                            eval.error_analysis.category.value,
                            eval.error_analysis.category_reasoning
                        ]
                    )

        

        score_filename = dir + "score_summary.csv"

        """if not os.path.exists(dir):
            os.makedirs(dir)"""

        with open(score_filename, 'w+', newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(
                [
                    "Subtask", "Total Items", "Total Score", "Average Score"
                ]
            )

            writer.writerow(
                [
                    "Vocabulary Awareness", score_summary["total_items"], score_summary["total_score"], score_summary["normalized_average"]
                ]
            )


        
       

        num_items = len(assessment_history)
        heatmap_data = np.full((num_items, 2), np.nan)

        heatmap_filename = dir + "per_pair_score_heatmap.png"

        for i, response in enumerate(assessment_history):
            for j, eval in enumerate(response.evaluations):
                heatmap_data[i, j] = eval.score.value

        cmap = sns.color_palette(["lightcoral", "lightgray", "mediumseagreen"])  # 0, NaN, 1
        bounds = [-0.5, 0.1, 0.9, 1.5]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, len(cmap))

        plt.figure(figsize=(6, num_items * 0.6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            cmap=cmap,
            norm=norm,
            cbar=False,
            xticklabels=["Pair 1", "Pair 2"],
            yticklabels=[f"{i+1}" for i in range(num_items)]
        )

        plt.title("VA Per-Pair Accuracy Heatmap")
        plt.xlabel("Pair Index")
        plt.ylabel("Item")
        plt.tight_layout()

        plt.savefig(heatmap_filename, dpi=300)
        plt.close()