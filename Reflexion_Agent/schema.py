from pydantic import BaseModel, Field
from typing import List, Optional

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")

class Answer_question(BaseModel):
    answer : str = Field(description="~80 word detailed answer to the question")
    search_queries : List[str] = Field(description="1-3 search queries for researching improvements to address the critique of current answer")
    reflection : Reflection = Field(description="your reflection on the initial answer")

class Revised_answer(Answer_question):
    """Revise your original answer to your question"""
    references : List[str] = Field(description="citations motivating your updated answer")