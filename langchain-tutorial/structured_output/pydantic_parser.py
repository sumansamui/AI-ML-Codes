from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(
        description="Name of a famous Indian cricketer"
    )

    age: int = Field(
        gt=18,
        description="Current age of the cricketer"
    )

    city: str = Field(
        description="City associated with the cricketer"
    )


model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.5
)


# Force the model output to follow the Person schema
structured_model = model.with_structured_output(Person)

result = structured_model.invoke("Mention one famous Indian cricketer.")

print(result)
print(type(result))

print("Name:", result.name)
print("Age:", result.age)
print("City:", result.city)