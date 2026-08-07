from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class ReviewAnalysis(BaseModel):
    pros: list[str] = Field(
        description="All positive points mentioned in the review"
    )

    cons: list[str] = Field(
        description="All negative points mentioned in the review"
    )

    sentiment: Literal["Positive", "Negative", "Mixed"] = Field(
        description=(
            "Overall sentiment. Use Mixed when the review contains "
            "both meaningful positive and negative opinions."
        )
    )


model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

structured_model = model.with_structured_output(ReviewAnalysis)

review = """
The smartphone has excellent battery life and a bright, smooth display.
The camera performs well in daylight, but low-light photos are average.
The phone also becomes slightly warm during long gaming sessions.
"""

result = structured_model.invoke(
    f"""
Analyze the following smartphone review.

Classify the sentiment as Mixed if both significant positive
and negative points are present.

{review}
"""
)

print("Pros:", result.pros)
print("Cons:", result.cons)
print("Sentiment:", result.sentiment)