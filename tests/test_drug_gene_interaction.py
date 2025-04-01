import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maf_tools.drug_gene_interactions import DrugGeneInteractionTool


def test_drug_gene_interaction():
    # Instantiate the tool
    tool = DrugGeneInteractionTool()

    # Test with a sample gene list
    genes = "BRAF,TP53"
    result = tool._run(genes)

    # Print the result
    print("Drug-Gene Interaction Results:")
    print(result)

if __name__ == "__main__":
    test_drug_gene_interaction()