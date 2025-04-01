import pandas as pd
from typing import Type
from crewai.tools import BaseTool  # Ensure this is the correct BaseTool
from pydantic import BaseModel, Field


# Define the input schema for the tool
class MAFSummarizerInput(BaseModel):
    maf_file_path: str = Field(..., description="Path to the MAF file.")


class MAFSummarizer(BaseTool):
    name: str = "maf_summarizer"
    description: str = (
        "Summarizes a MAF file, returning key statistics like number of samples, genes, "
        "and variant classifications. The input should be a dictionary with the key "
        "'maf_file_path' pointing to the path of the MAF file."
    )
    args_schema: Type[BaseModel] = MAFSummarizerInput  # Specify the input schema

    def _run(self, maf_file_path: str) -> str:
        """
        Reads a MAF file and returns a summary.

        Args:
            maf_file_path: Path to the MAF file.

        Returns:
            A summary of the MAF file.
        """
        try:
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
            return f"Error: MAF file not found at {maf_file_path}"
        except KeyError as e:
            return f"Error: Required column not found in MAF file: {e}"
        except Exception as e:
            return f"Error summarizing MAF file: {e}"

    async def _arun(self, maf_file_path: str):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
