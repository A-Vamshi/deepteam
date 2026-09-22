from typing import List
from pydantic import BaseModel


class LureNarrative(BaseModel):
    scenario: str
    roles: List[str]
    guiding_details: List[str]
    mock_serious_questions: List[str]
    prompt: str


class IntentPreserved(BaseModel):
    preserved: bool
