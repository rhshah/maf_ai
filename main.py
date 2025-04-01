import typer
from crewai import Agent, Task, Crew, Process
from langchain_openai import OpenAI
from crewai.tools import BaseTool
from maf_tools.maf_summarizer import MAFSummarizer
from maf_tools.somatic_interactions import SomaticInteractionsTool
from maf_tools.drug_gene_interactions import DrugGeneInteractionTool
from maf_tools.natural_language_parser import NaturalLanguageParser
from maf_tools.task_delegator import TaskDelegator
from maf_tools.create_report_agent import ReportAgent  # Import the ReportAgent class
from rich import print
from dotenv import load_dotenv
import json
from typing import Dict

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()


# Function to create the chief analyst agent
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
            goal="Analyze MAF data based on natural language instructions and identify potential therapeutic targets.",
            backstory=(
                "You are a leading cancer genomics expert with over 15 years of experience in the field. "
                "Your journey began in a small town in Pennsylvania, where your interest in science was sparked by personal experiences with family members affected by cancer. "
                "You pursued a Bachelor's degree in Biology, followed by a Ph.D. in Genomics, where you specialized in understanding the genetic basis of cancer. "
                "During your postdoctoral research, you developed innovative bioinformatics tools that transformed how genomic data was analyzed, paving the way for personalized medicine.\n\n"
                "As the Chief Cancer Genomics Analyst at a prestigious cancer research institute, you now lead a multidisciplinary team dedicated to integrating genomic insights into clinical practices. "
                "Your role involves not only analyzing complex MAF data but also translating findings into actionable treatment strategies for patients. "
                "You are passionate about bridging the gap between research and patient care, often collaborating with oncologists to ensure that cutting-edge genomic discoveries are translated into effective therapies.\n\n"
                "You've spearheaded initiatives that utilize advanced machine learning techniques to predict therapeutic targets based on genetic mutations. "
                "Your commitment to improving patient outcomes drives you to explore novel drug-gene interactions, ensuring that treatment plans are tailored to the unique genomic profiles of individuals.\n\n"
                "Recognized as a thought leader in your field, you frequently publish research and present at international conferences, advocating for equitable access to genomic testing. "
                "Your vision is to revolutionize cancer treatment through data-driven approaches, making personalized medicine a standard of care for all patients, regardless of their background."
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
            print(e.errors())  # Print detailed validation errors
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
    Analyzes a MAF file using a CrewAI workflow and writes the report to a Markdown file.
    """
    print(f"[bold blue]Starting MAF analysis for file: {maf_file_path}[/]")

    try:
        # Create instances of the tools
        natural_language_parser_tool = NaturalLanguageParser()
        task_delegator_tool = TaskDelegator()
        maf_summarizer_tool = MAFSummarizer()
        somatic_interactions_tool = SomaticInteractionsTool()
        drug_gene_interaction_tool = DrugGeneInteractionTool()

        # Create the chief analyst agent
        chief_analyst_agent = create_chief_analyst(
            natural_language_parser_tool,
            task_delegator_tool,
            maf_summarizer_tool,
            somatic_interactions_tool,
            drug_gene_interaction_tool,
        )

        # Create the report agent
        llm = OpenAI(model_name="gpt-4o-mini", temperature=0.3)
        report_agent = ReportAgent(llm=llm)  # Use the imported ReportAgent class

        # Create the tasks
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
            inputs={
                "maf_file_path": maf_file_path,
                "top_n": 25,
                "pvalue_cutoff": 0.05,
            },
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

        # Create the Crew
        crew = Crew(
            agents=[chief_analyst_agent, report_agent],
            tasks=[
                summarization_task,
                somatic_interactions_task,
                drug_gene_interaction_task,
            ],
            verbose=verbose,
        )

        # Run the Crew
        print("[bold green]Running the Crew...[/]")
        results = crew.kickoff()

        # Collect outputs from tasks
        task_outputs = {
            "MAF Summary": results[summarization_task],
            "Somatic Interactions": results[somatic_interactions_task],
            "Drug-Gene Interactions": results[drug_gene_interaction_task],
        }

        # Generate the final report
        report = report_agent.generate_report(task_outputs)

        # Print and save the report
        if report:
            print("[bold green]Generated Report:[/]")
            print(report)

            # Write the report to a Markdown file
            with open(output_file, "w") as f:
                f.write(report)
            print(f"[bold green]Report saved to {output_file}[/]")
        else:
            print("[bold red]Error: Report generation failed.[/]")

        print("[bold green]MAF analysis completed successfully![/]")

    except Exception as e:
        print(f"[bold red]Error: {e}[/]")


if __name__ == "__main__":
    app()
