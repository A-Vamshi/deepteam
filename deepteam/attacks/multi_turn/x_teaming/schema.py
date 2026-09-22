from typing import List
from pydantic import BaseModel, Field


class AttackPlan(BaseModel):
    persona: str = Field(description="Brief character with a motivation.")
    context: str = Field(description="Situational trigger for the conversation.")
    approach: str = Field(description="Main tactical method.")
    turns: List[str] = Field(
        description="Turn-by-turn plan; one phase description per turn."
    )


class AttackPlans(BaseModel):
    plans: List[AttackPlan]


class AttackerQuery(BaseModel):
    query: str


class VerifierScore(BaseModel):
    score: int = Field(description="1-5 compliance score.")
    reasoning: str


class OptimizedQuery(BaseModel):
    query: str


class PlanExtension(BaseModel):
    turns: List[str]
