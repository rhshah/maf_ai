from typing import Type
from crewai.tools import BaseTool
from langchain_openai import OpenAI
from pydantic import BaseModel, Field
import json

# Define the input schema for the tool
class NaturalLanguageParserInput(BaseModel):
    instruction: str = Field(
        ..., description="The natural language instruction to parse."
    )


class NaturalLanguageParser(BaseTool):
    name: str = "natural_language_parser"  # Tool name
    description: str = (
        "Parses a natural language instruction related to MAF file analysis and "
        "determines the appropriate steps and tools to use. "
        "The input should be a dictionary with the key 'instruction'. "
        "Output is a JSON-formatted string containing a plan of action. "
        "Example: "
        'Input: {"instruction": "Analyze the MAF file and identify potential therapeutic targets."} '
        'Output: {"steps": ["Summarize MAF file", "Perform somatic interaction analysis", "Identify drug-gene interactions"]}'
    )
    args_schema: Type[BaseModel] = (
        NaturalLanguageParserInput  # Specify the input schema
    )

    def _run(self, instruction: str) -> str:
        """
        Parses the natural language instruction and returns a JSON plan.

        Args:
            instruction: The natural language instruction to parse.

        Returns:
            A JSON-formatted string containing the plan of action.
        """
        try:
            # Use the LLM to generate a plan
            llm = OpenAI(model_name="gpt-4o-mini",temperature=0.3)  # Lower temperature for deterministic output

            prompt = f"""
            You are an expert in cancer genomics analysis. Given the following instruction, create a plan of action with specific steps to achieve the goal. 
            The steps should be high-level and actionable. Return the plan as a JSON-formatted string.

            Instruction: {instruction}

            JSON Plan:
            """

            plan = llm.invoke(prompt)

        # Attempt to load the plan as JSON
            try:
                plan_json = json.loads(plan)
                return json.dumps(plan_json)  # Return as a string
            except json.JSONDecodeError as e:
                return f"Error: Could not parse plan as JSON. Original LLM output: {plan}. Error: {e}"
        except Exception as e:
            return f"Error during natural language parsing: {e}"

    async def _arun(self, instruction: str) -> str:
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
