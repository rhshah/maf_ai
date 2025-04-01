import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maf_tools.maf_summarizer import MAFSummarizer

def test_maf_summarizer():
    summarizer = MAFSummarizer()
    inputs = {"maf_file_path": "/Users/shahr2/Downloads/msk_impact_2017/data_mutations.txt"}
    result = summarizer._run(inputs)
    print(result)

if __name__ == "__main__":
    test_maf_summarizer()