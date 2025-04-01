import requests
import pandas as pd
from typing import Type
from crewai.tools import BaseTool  # Ensure this is the correct BaseTool
from pydantic import BaseModel, Field


# Define the input schema for the tool
class DrugGeneInteractionInput(BaseModel):
    maf_file_path: str = Field(..., description="Path to the MAF file.")
    num_genes: int = Field(3, description="Number of top mutated genes to analyze.")
    num_interactions: int = Field(
        5, description="Number of top interactions to retrieve per gene."
    )


class DrugGeneInteractionTool(BaseTool):
    name: str = "drug_gene_interaction"
    description: str = (
        "Identifies potential therapeutic targets based on drug-gene interaction data from DGIdb. "
        "The input should include the path to a MAF file, the number of top mutated genes to analyze, "
        "and the number of top interactions to retrieve per gene."
    )
    args_schema: Type[BaseModel] = DrugGeneInteractionInput  # Specify the input schema

    def _run(self, maf_file_path: str, num_genes: int, num_interactions: int) -> str:
        """
        Identifies drug-gene interactions for the top mutated genes in a MAF file using a GraphQL query.

        Args:
            maf_file_path: Path to the MAF file.
            num_genes: Number of top mutated genes to analyze.
            num_interactions: Number of top interactions to retrieve per gene.

        Returns:
            A summary of drug-gene interactions for the specified genes.
        """
        try:
            # Read the MAF file
            maf_df = pd.read_csv(maf_file_path, sep="\t", comment="#")

            # Get the top mutated genes
            gene_counts = maf_df["Hugo_Symbol"].value_counts().nlargest(num_genes)
            top_genes = gene_counts.index.tolist()

            interactions = []

            for gene in top_genes:
                # GraphQL query
                query = """
                {
                  genes(names: ["%s"]) {
                    nodes {
                      interactions(first: %d) {
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
                """ % (gene, num_interactions)

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
                return "No drug-gene interactions found for the specified genes."

        except FileNotFoundError:
            return f"Error: MAF file not found at {maf_file_path}"
        except KeyError as e:
            return f"Error: Required column not found in MAF file: {e}"
        except Exception as e:
            raise RuntimeError(f"Error during drug-gene interaction analysis: {e}")

    async def _arun(self, maf_file_path: str, num_genes: int, num_interactions: int):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
