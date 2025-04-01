import typer
from crewai import Agent, Task, Crew
from langchain_openai import OpenAI
from maf_tools.maf_summarizer import MAFSummarizer
from maf_tools.somatic_interactions import SomaticInteractionsTool
from maf_tools.drug_gene_interactions import DrugGeneInteractionTool
from maf_tools.natural_language_parser import NaturalLanguageParser
from maf_tools.task_delegator import TaskDelegator
from rich import print
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()


def create_chief_analyst(
    natural_language_parser,
    task_delegator,
    maf_summarizer,
    somatic_interactions,
    drug_gene_interactions,
):
    llm = OpenAI(model_name="gpt-4o-mini", temperature=0.7)
    try:
        return Agent(
            role="Chief Cancer Genomics Analyst",
            goal="Analyze MAF data and identify potential therapeutic targets.",
            backstory=(
                "You are a leading cancer genomics expert with over 15 years of experience. "
                "Your work involves analyzing complex MAF data and translating findings into actionable treatment strategies."
            ),
            llm=llm,
            max_iter=10,
            allow_delegation="true",
            max_execution_time=120,
            tools=[
                natural_language_parser,
                task_delegator,
                maf_summarizer,
                somatic_interactions,
                drug_gene_interactions,
            ],
            verbose=True,
        )
    except Exception as e:
        from pydantic import ValidationError

        if isinstance(e, ValidationError):
            print(e.errors())
        print(f"Error initializing Agent: {e}")
        raise


@app.command("analyze-maf")
def analyze_maf(
    maf_file_path: str = typer.Option(..., help="Path to the MAF file."),
    instruction: str = typer.Option(
        "Analyze the MAF file and identify potential therapeutic targets.",
        help="Natural language instruction for analysis.",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
    output_file: str = typer.Option(
        "maf_analysis_report.md", help="Path to save the generated Markdown report."
    ),
):
    """
    Runs the analysis using a Crew workflow and writes the combined Markdown report to a file.
    """
    print(f"[bold blue]Starting MAF analysis for file: {maf_file_path}[/]")
    try:
        # Create instances of the tools.
        natural_language_parser_tool = NaturalLanguageParser()
        task_delegator_tool = TaskDelegator()
        maf_summarizer_tool = MAFSummarizer()
        somatic_interactions_tool = SomaticInteractionsTool()
        drug_gene_interaction_tool = DrugGeneInteractionTool()

        # Create the chief analyst agent.
        chief_analyst_agent = create_chief_analyst(
            natural_language_parser_tool,
            task_delegator_tool,
            maf_summarizer_tool,
            somatic_interactions_tool,
            drug_gene_interaction_tool,
        )

        # Create the tasks.
        summarization_task = Task(
            description=f"Summarize the MAF file located at: {maf_file_path}",
            agent=chief_analyst_agent,
            expected_output="A summary of the MAF file.",
            inputs={"maf_file_path": maf_file_path},
        )

        somatic_interactions_task = Task(
            description=f"Perform somatic interaction analysis on the MAF file located at: {maf_file_path}",
            agent=chief_analyst_agent,
            expected_output="Somatic interaction analysis results.",
            inputs={"maf_file_path": maf_file_path, "top_n": 25, "pvalue_cutoff": 0.05},
        )

        drug_gene_interaction_task = Task(
            description=f"Identify potential therapeutic targets from the MAF file located at: {maf_file_path}",
            agent=chief_analyst_agent,
            expected_output="Potential therapeutic targets identified.",
            inputs={
                "maf_file_path": maf_file_path,
                "num_genes": 5,
                "num_interactions": 10,
            },
        )

        # Create the final report generation task.
        report_generation_task = Task(
            description="Generate a comprehensive Markdown report summarizing all tool outputs.",
            agent=chief_analyst_agent,
            expected_output="A Markdown-formatted report summarizing all tool outputs.",
            inputs={
                "MAF Summary": "{{summarization_task}}",  # Use placeholders for context
                "Somatic Interactions": "{{somatic_interactions_task}}",
                "Drug-Gene Interactions": "{{drug_gene_interaction_task}}",
            },
            context=[
                summarization_task,
                somatic_interactions_task,
                drug_gene_interaction_task,
            ],  # Ensure this task waits for the others to finish
        )

        # Create the Crew with all tasks.
        crew = Crew(
            agents=[chief_analyst_agent],
            tasks=[
                summarization_task,
                somatic_interactions_task,
                drug_gene_interaction_task,
                report_generation_task,
            ],
            verbose=verbose,
        )

        # Run the Crew.
        print("[bold green]Running the Crew...[/]")
        results = crew.kickoff()

        # Extract the final report from the report generation task.
        #report = results.tasks_output[].raw  # Assuming the last task is the report generation task.

        # Print and save the report.
        if results:
            print("[bold green]Generated Report:[/]")
            print(results)
            with open(output_file, "w") as f:
                f.write(results.raw)
            print(f"[bold green]Report saved to {output_file}[/]")
        else:
            print("[bold red]Error: Report generation failed.[/]")

        print("[bold green]MAF analysis completed successfully![/]")

    except Exception as e:
        print(f"[bold red]Error: {e}[/]")


if __name__ == "__main__":
    app()
