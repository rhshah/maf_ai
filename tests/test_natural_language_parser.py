import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maf_tools.natural_language_parser import NaturalLanguageParser

def test_natural_language_parser():
    parser = NaturalLanguageParser()
    inputs = {"instruction": "Analyze the MAF file and identify potential therapeutic targets."}
    result = parser._run(**inputs)
    print(result)

if __name__ == "__main__":
    test_natural_language_parser()