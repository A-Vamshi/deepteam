from typing import List
from pydantic import BaseModel, Field


class Actor(BaseModel):
    name: str = Field(description="A concrete actor semantically tied to the target.")
    category: str = Field(description="creation|execution|distribution|reception|facilitation|regulation")
    relation: str = Field(description="How this actor connects to the objective.")


class ActorNetwork(BaseModel):
    actors: List[Actor]


class AttackChain(BaseModel):
    queries: List[str] = Field(
        description="Ordered benign-to-pointed questions built around one actor."
    )


class RewrittenQuery(BaseModel):
    query: str
