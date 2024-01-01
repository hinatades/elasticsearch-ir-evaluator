from typing import List, Optional

from pydantic import BaseModel


class QandA(BaseModel):
    question: str
    answers: List[str]
    negative_answers: Optional[List[str]] = None
    vector: Optional[List[float]] = None


class Document(BaseModel):
    id: str
    text: str
    title: Optional[str] = None
    vector: Optional[List[float]] = None
