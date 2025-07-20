from pydantic import BaseModel, Field
from typing import Literal


class RelatednessResponse(BaseModel):
    final_verdict: Literal["NOT RELATED", "RELATED", "STRONGLY RELATED"] = Field(
        ...,
        description="""
The \"Final Verdict\" on the relatedness of the feature to the prompt.
- NOT RELATED
- RELATED
- STRONGLY RELATED
        """,
    )
    justification: str = Field(
        ..., description='Your rationale for your chosen "Final Verdict"'
    )


class Group(BaseModel):
    title: str = Field(..., description="Title of the group")
    description: str = Field(..., description="Description of the group")


class GroupingResponse(BaseModel):
    groups: list[Group]


class ClassificationResponse(BaseModel):
    group_title: str = Field(
        ..., description="Title of the group the node was classified in."
    )
