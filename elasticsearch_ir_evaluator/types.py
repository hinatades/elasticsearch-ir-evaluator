from typing import List, Optional

from pydantic import BaseModel, field_validator


class QandA(BaseModel):
    question: str
    answers: List[str]
    negative_answers: Optional[List[str]] = None
    vector: Optional[List[float]] = None


class Passage(BaseModel):
    text: Optional[str] = None
    vector: Optional[List[float]] = None


class Document(BaseModel):
    id: str
    text: Optional[str] = None
    title: Optional[str] = None
    vector: Optional[List[float]] = None
    passages: Optional[List[Passage]] = None


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

    def to_markdown(self):
        headers = ["Metric", "Value"]
        rows = [
            ("Precision", self.Precision),
            ("Recall", self.Recall),
            ("MRR", self.MRR),
            ("MAP", self.MAP),
            ("CG", self.CG),
            ("nDCG", self.nDCG),
            ("FPR", self.FPR),
            ("BPref", self.BPref),
        ]
        # Determine the maximum width of each column
        max_widths = [
            max(len(str(row[i])) for row in rows + [headers]) for i in range(2)
        ]
        # Create a format specifier for each column
        column_formats = [f"{{:<{max_width}}}" for max_width in max_widths]
        # Format header
        header_line = (
            "| "
            + " | ".join(
                column_formats[i].format(header) for i, header in enumerate(headers)
            )
            + " |"
        )
        separator_line = (
            "|-" + "-|-".join("-" * max_width for max_width in max_widths) + "-|"
        )
        # Format rows
        row_lines = [
            "| "
            + " | ".join(
                column_formats[i].format(row[i] if row[i] is not None else "")
                for i in range(2)
            )
            + " |"
            for row in rows
        ]
        # Combine all parts
        return "\n".join([header_line, separator_line] + row_lines)
