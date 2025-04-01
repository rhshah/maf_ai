import requests
import pandas as pd
from typing import Type
from crewai.tools import BaseTool  # Ensure this is the correct BaseTool
from pydantic import BaseModel, Field


# Define the input schema for the tool
class DrugGeneInteractionInput(BaseModel):
    maf_file_path: str = Field(..., description="Path to the MAF file.")


class DrugGeneInteractionTool(BaseTool):
    name: str = "drug_gene_interaction"
    description: str = (
        "Identifies potential therapeutic targets based on drug-gene interaction data from DGIdb. "
        "The input should be the path to a MAF file. "
        "The tool analyzes the top 3 most frequently mutated genes in the MAF file."
    )
    args_schema: Type[BaseModel] = DrugGeneInteractionInput  # Specify the input schema

    def _run(self, maf_file_path: str) -> str:
        """
        Identifies drug-gene interactions for the top 3 genes in a MAF file using a GraphQL query.

        Args:
            maf_file_path: Path to the MAF file.

        Returns:
            A summary of drug-gene interactions for the top 3 genes.
        """
        try:
            # Read the MAF file
            maf_df = pd.read_csv(maf_file_path, sep="\t", comment="#")

            # Get the top 3 most frequently mutated genes
            gene_counts = maf_df["Hugo_Symbol"].value_counts().nlargest(3)
            top_genes = gene_counts.index.tolist()

            interactions = []

            for gene in top_genes:
                # GraphQL query
                query = (
                    """
                {
                  genes(names: ["%s"]) {
                    nodes {
                      interactions {
                        drug {
                          name
                          conceptId
                        }
                        interactionScore
                        interactionTypes {
                          type
                          directionality
                        }
                        interactionAttributes {
                          name
                          value
                        }
                        publications {
                          pmid
                        }
                        sources {
                          sourceDbName
                        }
                      }
                    }
                  }
                }
                """
                    % gene
                )

                # Send the GraphQL request
                url = "https://dgidb.org/api/graphql"
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, json={"query": query}, headers=headers)

                if response.status_code == 200:
                    data = response.json()

                    # Validate API response structure
                    if "data" in data and data["data"]["genes"]["nodes"]:
                        for interaction in data["data"]["genes"]["nodes"][0].get(
                            "interactions", []
                        ):
                            drug_name = interaction["drug"]["name"]
                            interaction_types = [
                                f"{t['type']} ({t['directionality']})"
                                for t in interaction.get("interactionTypes", [])
                            ]
                            sources = ", ".join(
                                source["sourceDbName"]
                                for source in interaction.get("sources", [])
                            )
                            interactions.append(
                                f"{gene}: {drug_name} - Types: {', '.join(interaction_types)} - Sources: {sources}"
                            )
                    else:
                        interactions.append(f"{gene}: No interactions found.")
                else:
                    interactions.append(
                        f"{gene}: Error retrieving data from DGIdb. HTTP Status Code: {response.status_code}"
                    )

            if interactions:
                return "\n".join(interactions)
            else:
                return "No drug-gene interactions found for the top 3 genes."

        except FileNotFoundError:
            return f"Error: MAF file not found at {maf_file_path}"
        except KeyError as e:
            return f"Error: Required column not found in MAF file: {e}"
        except Exception as e:
            raise RuntimeError(f"Error during drug-gene interaction analysis: {e}")

    async def _arun(self, maf_file_path: str):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
