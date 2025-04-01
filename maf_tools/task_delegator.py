from langchain.tools import BaseTool
import json


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

    def __init__(self):
        super().__init__()

    def _run(self, inputs: dict) -> str:
        """
        Delegates tasks to other agents based on the plan.

        Args:
            inputs: A dictionary with the following keys:
                - plan_json: A JSON-formatted string containing the plan.
                - maf_file_path: Path to the MAF file.

        Returns:
            A string representation of the delegated tasks.
        """
        try:
            # Validate input
            if not isinstance(inputs, dict):
                raise ValueError("Input must be a dictionary.")
            if "plan_json" not in inputs or "maf_file_path" not in inputs:
                raise ValueError(
                    "Input dictionary must contain the keys 'plan_json' and 'maf_file_path'."
                )

            plan_json = inputs["plan_json"]
            maf_file_path = inputs["maf_file_path"]

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

    async def _arun(self, inputs: dict):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
