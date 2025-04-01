from crewai import Task
from typing import Dict


class ReportGenerationTask(Task):
    def __init__(self, description: str, inputs: Dict[str, str], **kwargs):
        super().__init__(
            description=description,
            expected_output="A comprehensive Markdown report summarizing all tool outputs.",
            **kwargs,
        )
        self.inputs = inputs

    def _run(self) -> str:
        """
        Generates a comprehensive Markdown report from the outputs of various tools.
        """
        try:
            # Extract inputs
            maf_summary = self.inputs.get("MAF Summary", "No MAF summary available.")
            somatic_interactions = self.inputs.get(
                "Somatic Interactions", "No somatic interactions available."
            )
            drug_gene_interactions = self.inputs.get(
                "Drug-Gene Interactions", "No drug-gene interactions available."
            )

            # Generate the report
            report = "# Comprehensive MAF Analysis Report\n\n"

            # Add MAF Summary
            report += "## MAF Summary\n\n"
            report += f"```\n{maf_summary}\n```\n\n"

            # Add Somatic Interactions
            report += "## Somatic Interactions\n\n"
            report += "| Gene1 | Gene2 | pValue |\n"
            report += "|-------|-------|--------|\n"
            for line in somatic_interactions.split("\n"):
                if line.strip():
                    report += f"| {line.replace(',', ' | ')} |\n"
            report += "\n"

            # Add Drug-Gene Interactions
            report += "## Drug-Gene Interactions\n\n"
            report += "| Gene | Drug | Interaction Type | Sources |\n"
            report += "|------|------|------------------|---------|\n"
            for line in drug_gene_interactions.split("\n"):
                if line.strip():
                    report += f"| {line.replace(',', ' | ')} |\n"
            report += "\n"

            # Add Conclusion
            report += "## Conclusion\n\n"
            report += (
                "This report summarizes the results of the MAF analysis, including the MAF file summary, "
                "somatic interaction analysis, and drug-gene interactions. The findings provide valuable insights "
                "into potential therapeutic targets and their clinical relevance.\n"
            )

            return report
        except Exception as e:
            return f"Error generating report: {e}"