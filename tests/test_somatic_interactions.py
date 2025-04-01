import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maf_tools.somatic_interactions import SomaticInteractionsTool

def test_somatic_interactions():
    tool = SomaticInteractionsTool()
    inputs = {
        "maf_file_path": "/Users/shahr2/Downloads/msk_impact_2017/data_mutations.txt",
        "top_n": 25,
        "pvalue_cutoff": 0.05,
    }
    result = tool._run(inputs)
    print(result)

if __name__ == "__main__":
    test_somatic_interactions()