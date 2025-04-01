from typing import Type
from crewai.tools import BaseTool  # Ensure this is the correct BaseTool
from pydantic import BaseModel, Field
import json


# Define the input schema for the tool
class TaskDelegatorInput(BaseModel):
    plan_json: str = Field(
        ..., description="A JSON-formatted string containing the plan."
    )
    maf_file_path: str = Field(..., description="Path to the MAF file.")


class TaskDelegator(BaseTool):
    name: str = "task_delegator"
    description: str = (
        "Delegates tasks to other agents based on a plan of action. "
        "The input should be a dictionary with the following keys: "
        "'plan_json' (a JSON-formatted string containing the plan) and "
        "'maf_file_path' (path to the MAF file). "
        "The plan should have a 'steps' key, where steps are high-level actions. "
        "This tool returns a list of tasks that have been delegated."
    )
    args_schema: Type[BaseModel] = TaskDelegatorInput  # Specify the input schema

    def _run(self, plan_json: str, maf_file_path: str) -> str:
        """
        Delegates tasks to other agents based on the plan.

        Args:
            plan_json: A JSON-formatted string containing the plan.
            maf_file_path: Path to the MAF file.

        Returns:
            A string representation of the delegated tasks.
        """
        try:
            # Parse the plan JSON
            plan = json.loads(plan_json)
            steps = plan.get("steps", [])

            delegated_tasks = []
            for step in steps:
                if "Summarize MAF file" in step:
                    delegated_tasks.append(
                        f"Summarize the MAF file located at: {maf_file_path}"
                    )
                elif "Perform somatic interaction analysis" in step:
                    delegated_tasks.append(
                        f"Perform somatic interaction analysis on the MAF file located at: {maf_file_path}"
                    )
                elif "Identify drug-gene interactions" in step:
                    delegated_tasks.append(
                        f"Identify potential therapeutic targets from the MAF file located at: {maf_file_path}"
                    )
                else:
                    delegated_tasks.append(f"Unknown Task: {step}")

            return ", ".join(delegated_tasks)  # Return a string of tasks
        except json.JSONDecodeError as e:
            return f"Error: Could not parse plan as JSON. Error: {e}"
        except Exception as e:
            return f"Error during task delegation: {e}"

    async def _arun(self, plan_json: str, maf_file_path: str):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
