import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maf_tools.task_delegator import TaskDelegator

def test_task_delegator():
    delegator = TaskDelegator()
    inputs = {
        "plan_json": '{"steps": ["Summarize MAF file", "Perform somatic interaction analysis", "Identify drug-gene interactions"]}',
        "maf_file_path": "/Users/shahr2/Downloads/msk_impact_2017/data_mutations.txt",
    }
    result = delegator._run(inputs)
    print(result)

if __name__ == "__main__":
    test_task_delegator()