from crewai import Agent
from typing import Type
from crewai.tools import BaseTool
from langchain_openai import OpenAI
from pydantic import BaseModel, Field


class MyToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "What this tool does. It's vital for effective utilization."
    args_schema: Type[BaseModel] = MyToolInput

    def _run(self, argument: str) -> str:
        # Your tool's logic here
        return "Tool's result"


# Instantiate the tool
dummy_tool = MyCustomTool()
print(f"dummy tool is instance of BaseTool: {isinstance(dummy_tool, BaseTool)}")

# Instantiate the LLM
llm = OpenAI(temperature=0.7)

# Test the Agent initialization
try:
    agent = Agent(
        role="Test Agent",
        goal="Test the Agent with a single tool.",
        backstory="Testing backstory.",
        memory=None,
        tools=[dummy_tool],  # Pass the tool instance
        llm=llm,
        verbose=True,
    )
    print("Agent initialized successfully.")
except Exception as e:
    from pydantic import ValidationError
    if isinstance(e, ValidationError):
        print(e.errors())  # Print detailed validation errors
    print(f"Error initializing Agent: {e}")