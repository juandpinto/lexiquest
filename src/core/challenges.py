from abc import abstractmethod
from typing import ClassVar, Dict, Type, Any, List, Tuple
from typing_extensions import Self
from pydantic import BaseModel, Field


class BaseChallenge(BaseModel):
    challenge_type: str = Field(..., description="The type of the challenge")

    _registry: ClassVar[Dict[str, Type["BaseChallenge"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "type_key"):
            BaseChallenge._registry[cls.type_key] = cls

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseChallenge":
        type_key = data.get("challenge_type")
        type_key = 'triplet' if type_key == 'Vocabulary Awareness' else type_key
        if type_key not in cls._registry:
            raise ValueError(f"Unknown challenge type: {type_key}")
        return cls._registry[type_key](**data)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def get_class_by_type(cls, type_key: str) -> type["BaseChallenge"]:
        return cls._registry[type_key]

    @classmethod
    def get_example_for(cls, type_key: str) -> dict:
        if type_key not in cls._registry:
            raise ValueError(f"Unknown challenge type: {type_key}")
        return cls._registry[type_key].example().model_dump()

    @classmethod
    @abstractmethod
    def example(cls) -> "BaseChallenge":
        """Return a dummy example instance"""
        raise NotImplementedError

    def summarize(self) -> "BaseChallenge":
        """Use this method to resolve a base challenge to its specific type"""
        raise NotImplementedError


class Pairing(BaseModel):
    words: List[str] = Field(description="List of two words that are associated")
    justification: str = Field(default="String representing the justification for the pairing")


class PhonemicAwareness(BaseChallenge):
    type_key: str = 'phonemic'
    challenge_type: str = 'phonemic awareness'

    non_word_pair: Tuple[str, str] = Field(description="Tuple containing non-word single consonant removal pairs")
    phonemic_pair: Tuple[str, str] = Field(description="Tuple containing phonetic spelling of non-word pairs")

    def summarize(self) -> Self:
        return self

    @classmethod
    def class_type(cls) -> type["PhonemicAwareness"]:
        return cls

    @classmethod
    def example(cls) -> "PhonemicAwareness":
        return cls(
            non_word_pair=("bip", "ip"),
            phonemic_pair=("bɪp", "ɪp")
        )


class InferentialVocabulary(BaseChallenge):
    type_key: str = "Inferential Vocabulary"
    challenge_type: str = "IV"

    a_question: str = Field(description="String representing the primary inferential vocabulary question")
    b_question: str = Field(description="String representation of alternative inferential vocabulary question")
    word_meaning_pair: Tuple[str, str] = Field(description="A tuple containing a word and its expected meaning")

    def summarize(self) -> Self:
        return self

    @classmethod
    def class_type(cls) -> type["BaseChallenge"]:
        return cls

    @classmethod
    def example(cls) -> "InferentialVocabulary":
        return cls(
            a_question="Scott was tumbling off his skateboard. He kept getting hurt. What does tumble mean?",
            b_question="Does tumble mean to ride or to fall?",
            word_meaning_pair=("tumble", "to fall suddenly, clumsily, or headlong")
        )


class ChallengeTriplet(BaseChallenge):
    type_key: str = "triplet"
    challenge_type: str = "triplet"

    triplet: List[str] = Field(description="List of three words for association")
    pairings: List[Pairing] = Field(description="List of pairings with their associated definitions")

    def summarize(self) -> Self:
        return self

    @classmethod
    def class_type(cls) -> type["BaseChallenge"]:
        return cls

    @classmethod
    def example(cls) -> "ChallengeTriplet":
        return cls(
            triplet=["dog", "cat", "bone"],
            pairings=[
                {"words": ["dog", "cat"], "justification": "because they are both animals"},
                {"words": ["dog", "bone"], "justification": "because dogs like bones"}
            ]
        )
