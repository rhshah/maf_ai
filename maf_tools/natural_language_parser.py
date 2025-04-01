from langchain.tools import BaseTool
from langchain_openai import OpenAI
import json


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

    def __init__(self):
        super().__init__()

    def _run(self, inputs: dict) -> str:
        """
        Parses the natural language instruction and returns a JSON plan.

        Args:
            inputs: A dictionary with the key 'instruction'.

        Returns:
            A JSON-formatted string containing the plan of action.
        """
        try:
            # Validate input
            if not isinstance(inputs, dict):
                raise ValueError("Input must be a dictionary.")
            if "instruction" not in inputs:
                raise ValueError("Input dictionary must contain the key 'instruction'.")

            instruction = inputs["instruction"]

            # Use the LLM to generate a plan
            llm = OpenAI(temperature=0.3)  # Lower temperature for deterministic output

            prompt = f"""
            You are an expert in cancer genomics analysis. Given the following instruction, create a plan of action with specific steps to achieve the goal. The steps should be high-level and actionable. Return the plan as a JSON-formatted string.

            Instruction: {instruction}

            JSON Plan:
            """

            plan = llm.invoke(prompt)

            # Attempt to load the plan as JSON
            plan_json = json.loads(plan)
            return json.dumps(plan_json)  # Return as a string

        except json.JSONDecodeError as e:
            return f"Error: Could not parse plan as JSON. Original LLM output: {plan}. Error: {e}"
        except Exception as e:
            return f"Error during natural language parsing: {e}"

    async def _arun(self, inputs: dict) -> str:
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
