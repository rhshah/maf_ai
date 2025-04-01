import pandas as pd
from langchain.tools import BaseTool


class MAFSummarizer(BaseTool):
    name: str = "maf_summarizer"
    description: str = (
        "Summarizes a MAF file, returning key statistics like number of samples, genes, "
        "and variant classifications. The input should be a dictionary with the key "
        "'maf_file_path' pointing to the path of the MAF file."
    )

    def __init__(self):
        super().__init__()

    def _run(self, inputs: dict) -> str:
        """
        Reads a MAF file and returns a summary.
        The input should be a dictionary with the key 'maf_file_path'.
        """
        try:
            # Validate input
            if not isinstance(inputs, dict) or "maf_file_path" not in inputs:
                raise ValueError(
                    "Input must be a dictionary with a 'maf_file_path' key."
                )

            maf_file_path = inputs["maf_file_path"]

            # Read the MAF file
            maf_df = pd.read_csv(
                maf_file_path, sep="\t", comment="#"
            )  # Adjust separator and comment character if needed

            # Calculate statistics
            sample_count = maf_df["Tumor_Sample_Barcode"].nunique()
            gene_count = maf_df["Hugo_Symbol"].nunique()
            variant_classifications = (
                maf_df["Variant_Classification"].value_counts().to_dict()
            )

            # Create summary
            summary = (
                f"MAF Summary:\n"
                f"  Number of Samples: {sample_count}\n"
                f"  Number of Genes: {gene_count}\n"
                f"  Variant Classifications: {variant_classifications}"
            )
            return summary

        except FileNotFoundError:
            return f"Error: MAF file not found at {inputs.get('maf_file_path', 'Unknown Path')}"
        except KeyError as e:
            return f"Error: Required column not found in MAF file: {e}"
        except Exception as e:
            return f"Error summarizing MAF file: {e}"

    async def _arun(self, inputs: dict):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
