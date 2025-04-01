import requests
from langchain.tools import BaseTool


class DrugGeneInteractionTool(BaseTool):
    name: str = "drug_gene_interaction"
    description: str = (
        "Identifies potential therapeutic targets based on drug-gene interaction data from DGIdb. "
        "The input should be a list of genes (Hugo Symbols) separated by commas. "
        "Output is a summary of drug-gene interactions."
    )

    def __init__(self):
        super().__init__()

    def _run(self, genes: str) -> str:
        """
        Identifies drug-gene interactions using a GraphQL query.
        """
        try:
            gene_list = [g.strip() for g in genes.split(",")]
            interactions = []

            for gene in gene_list:
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
                return "No drug-gene interactions found for the given genes."

        except Exception as e:
            raise RuntimeError(f"Error during drug-gene interaction analysis: {e}")

    async def _arun(self, genes: str):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution")
