"""Output schema: every pipeline result must validate against this."""
from pydantic import BaseModel, Field


class Medication(BaseModel):
    name: str
    dosage: str | None = None


class ClinicalNote(BaseModel):
    diagnosis: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)
    summary: str = ""
