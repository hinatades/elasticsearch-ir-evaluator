from typing import List, Optional

from pydantic import BaseModel, validator


class QandA(BaseModel):
    question: str
    answers: List[str]
    negative_answers: Optional[List[str]] = None
    vector: Optional[List[float]] = None


class Passage(BaseModel):
    text: str
    vector: Optional[List[float]] = None


class Document(BaseModel):
    id: str
    text: Optional[str] = None
    title: Optional[str] = None
    vector: Optional[List[float]] = None
    passages: Optional[List[Passage]] = None

    @validator("passages", pre=True, always=True)
    def check_text_or_passages(cls, passages, values, **kwargs):
        if values.get("text") is None and (
            not passages or all(p.text is None for p in passages)
        ):
            raise ValueError(
                'Either "text" or at least one "passages[].text" must be provided'
            )
        return passages
