from typing import Type
from crewai.tools import BaseTool  # Ensure this is the correct BaseTool
from pydantic import BaseModel, Field
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests
import pandas as pd


# Define the input schema for the tool
class SomaticInteractionsInput(BaseModel):
    maf_file_path: str = Field(..., description="Path to the MAF file.")
    top_n: int = Field(25, description="Number of top mutated genes to consider.")
    pvalue_cutoff: float = Field(0.05, description="P-value cutoff for significance.")


class SomaticInteractionsTool(BaseTool):
    name: str = "somatic_interactions"
    description: str = (
        "Identifies mutually exclusive or co-occurring gene sets in a MAF file using Fisher's Exact Test. "
        "The input should include 'maf_file_path' (path to the MAF file), 'top_n' (number of top mutated genes to consider), "
        "and 'pvalue_cutoff' (p-value cutoff for significance)."
    )
    args_schema: Type[BaseModel] = SomaticInteractionsInput  # Specify the input schema

    def _run(self, maf_file_path: str, top_n: int, pvalue_cutoff: float) -> str:
        """
        Analyzes somatic interactions in a MAF file.

        Args:
            maf_file_path: Path to the MAF file.
            top_n: Number of top mutated genes to consider.
            pvalue_cutoff: The p-value cutoff for significance.

        Returns:
            A string representation of the results (gene pairs, p-values, etc.).
        """
        try:
            # Read the MAF file
            maf_df = pd.read_csv(
                maf_file_path, sep="\t", comment="#"
            )  # Adjust separator if needed
            sample_ids = maf_df["Tumor_Sample_Barcode"].unique()

            # 1. Gene Selection
            gene_counts = maf_df["Hugo_Symbol"].value_counts().nlargest(top_n)
            top_genes = gene_counts.index.tolist()

            results = []

            # 2. Pairwise Iteration
            for i in range(len(top_genes)):
                for j in range(i + 1, len(top_genes)):
                    gene1 = top_genes[i]
                    gene2 = top_genes[j]

                    # 3. Contingency Table Creation
                    gene1_mutated = maf_df["Tumor_Sample_Barcode"][
                        maf_df["Hugo_Symbol"] == gene1
                    ].unique()
                    gene2_mutated = maf_df["Tumor_Sample_Barcode"][
                        maf_df["Hugo_Symbol"] == gene2
                    ].unique()

                    n11 = len(set(gene1_mutated) & set(gene2_mutated))  # Both mutated
                    n10 = len(set(gene1_mutated) - set(gene2_mutated))  # Gene1 only
                    n01 = len(set(gene2_mutated) - set(gene1_mutated))  # Gene2 only
                    n00 = len(sample_ids) - n11 - n10 - n01  # Neither

                    # 4. Fisher's Exact Test
                    contingency_table = [[n11, n10], [n01, n00]]
                    oddsratio, pvalue = fisher_exact(contingency_table)

                    # Determine event type (Co-occurrence or Mutually Exclusive)
                    if oddsratio > 1:
                        event = "Co_Occurence"
                    else:
                        event = "Mutually_Exclusive"

                    results.append(
                        [gene1, gene2, pvalue, oddsratio, n00, n01, n11, n10, event]
                    )

            # Create a DataFrame from the results
            results_df = pd.DataFrame(
                results,
                columns=[
                    "gene1",
                    "gene2",
                    "pValue",
                    "oddsRatio",
                    "00",
                    "01",
                    "11",
                    "10",
                    "Event",
                ],
            )

            # 5. P-value Adjustment (Benjamini-Hochberg)
            reject, pvals_corrected, _, _ = multipletests(
                results_df["pValue"], method="fdr_bh"
            )
            results_df["pAdjust"] = pvals_corrected

            # Filter based on p-value cutoff
            significant_interactions = results_df[results_df["pAdjust"] < pvalue_cutoff]

            # Format the output as a string
            if significant_interactions.empty:
                return "No significant somatic interactions found."

            return significant_interactions.to_string()

        except FileNotFoundError:
            return f"Error: MAF file not found at {maf_file_path}"
        except KeyError as e:
            return f"Error: Required column not found in MAF file: {e}"
        except Exception as e:
            return f"Error during somatic interaction analysis: {e}"

    async def _arun(self, maf_file_path: str, top_n: int, pvalue_cutoff: float):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
