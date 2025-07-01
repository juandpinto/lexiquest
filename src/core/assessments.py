from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from core.challenges import Pairing




class PairingScoreEnum(int, Enum):
    incorrect = 0
    correct = 1


class ItemScoreEnum(int, Enum):
    zero = 0
    one = 1
    two = 2




class ChallengePairings(BaseModel):
    pairings: List[Pairing] = Field(description="List of word pairs with their associated justification.")



class PairingEvaluation(BaseModel):
    evaluated_pairing: Pairing = Field(
        description="The selected word pair along with the student's justification."
    )
    
    pair_is_valid: bool = Field(
        description="True if the selected pair reflects a valid semantic relationship."
    )
    
    justification_is_valid: bool = Field(
        description="True if the justification meaningfully and clearly supports the selected pair."
    )
    
    score: PairingScoreEnum = Field(
        description="Numerical score for this response: 1 = correct, if both word pair AND justification are correct; 0 = incorrect, if word pair and/or justification is incorrect."
    )



class ResponseEvaluation(BaseModel):
    evaluations: List[PairingEvaluation] = Field(
        description="List of evaluations for each word pair and their associated justification."
    )

    total_score: Optional[ItemScoreEnum] = Field(
        default=None,
        description="Total score for this item, computed as the sum of scores for each evaluated pair. Range: 0–2."
    )
    
    def update_total_score(self):
        total = sum(eval.score.value for eval in self.evaluations)
        self.total_score = ItemScoreEnum(total)