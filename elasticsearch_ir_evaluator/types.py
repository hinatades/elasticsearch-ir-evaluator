from typing import List, Optional

from pydantic import BaseModel, field_validator


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

    @field_validator("passages")
    def check_text_or_passages(cls, passages, values, **kwargs):
        if values.get("text") is None and (
            not passages or all(p.text is None for p in passages)
        ):
            raise ValueError(
                'Either "text" or at least one "passages[].text" must be provided'
            )
        return passages


class Result(BaseModel):
    Precision: Optional[float] = None
    Recall: Optional[float] = None
    FPR: Optional[float] = None
    nDCG: Optional[float] = None
    MAP: Optional[float] = None
    CG: Optional[float] = None
    BPref: Optional[float] = None
    MRR: Optional[float] = None

    @field_validator(
        "Precision",
        "Recall",
        "MRR",
        "CG",
        "nDCG",
        "MAP",
        "FPR",
        "BPref",
    )
    def round_float(cls, v):
        return round(v, 3) if v is not None else None
