from pydantic import BaseModel, Field

# Pydantic for zs or fs
class Class(BaseModel):
    result: str = Field(description="(A)Hate or (B)Not Hate")


# Pydantic for RAG
class RAG_Analyse_Class(BaseModel):
    similar_example: str = Field(description="List the most relevant example from the Reference Examples to the Input Post, highlighting the aspects that make it helpful for classification.")
    explanation: str = Field(description="Explain why you classified it as explicit_hate, implicit_hate, or not_hate, referencing the similar example.")
    result: str = Field(description="Choose one of the following labels: explicit_hate, implicit_hate or not_hate")

