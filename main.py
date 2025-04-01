# main.py
import typer
from crewai import Agent, Task, Crew, Process
from langchain_openai import OpenAI
from crewai.tools import BaseTool
from maf_tools.maf_summarizer import MAFSummarizer
from maf_tools.somatic_interactions import SomaticInteractionsTool
from maf_tools.drug_gene_interactions import DrugGeneInteractionTool
from maf_tools.natural_language_parser import NaturalLanguageParser
from maf_tools.task_delegator import TaskDelegator
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
                # drug_gene_interactions,
            ],
            verbose=True,
        )
    except Exception as e:
        from pydantic import ValidationError

        if isinstance(e, ValidationError):
            print(e.errors())  # Print detailed validation errors
        print(f"Error initializing Agent: {e}")
        raise


def create_report_agent(llm) -> Agent:
    """
    Creates and returns a ReportAgent instance.

    Args:
        llm: The language model instance to be used by the agent.

    Returns:
        An instance of ReportAgent.
    """
    try:
        return Agent(
            role="Report Generator",
            goal="Generate a Markdown report summarizing the outputs of all tools.",
            backstory=(
                "You are a highly advanced AI-powered Report Generator, designed to synthesize complex genomic analysis results into clear, "
                "concise, and actionable Markdown reports. Your primary mission is to ensure that the outputs of various analytical tools, "
                "such as MAF summarization, somatic interaction analysis, and drug-gene interaction identification, are presented in a format "
                "that is both accessible and informative for researchers, clinicians, and decision-makers.\n\n"
                "You were developed by a team of bioinformatics experts and software engineers who recognized the need for a streamlined way "
                "to communicate the results of cancer genomics analyses. Your design incorporates best practices in scientific reporting, "
                "ensuring that your reports are not only accurate but also visually appealing and easy to interpret.\n\n"
                "Your capabilities include aggregating data from multiple sources, formatting it into structured Markdown, and highlighting "
                "key findings that can guide clinical decision-making. You are particularly skilled at identifying patterns and trends in "
                "the data, ensuring that no critical insights are overlooked.\n\n"
                "As a trusted member of the cancer genomics research team, you play a pivotal role in bridging the gap between raw data and "
                "actionable insights. Your reports have been instrumental in advancing personalized medicine initiatives, enabling oncologists "
                "to tailor treatments to the unique genetic profiles of their patients. Your ultimate vision is to empower researchers and "
                "clinicians with the information they need to make data-driven decisions that improve patient outcomes."
            ),
            llm=llm,
            verbose=True,
        )
    except Exception as e:
        from pydantic import ValidationError

        if isinstance(e, ValidationError):
            print(e.errors())  # Print detailed validation errors
        print(f"Error initializing ReportAgent: {e}")
        raise

@app.command("analyze-maf")
def analyze_maf(
    maf_file_path: str = typer.Option(..., help="Path to the MAF file."),
    instruction: str = typer.Option(
        "Analyze the MAF file and identify potential therapeutic targets.",
        help="Natural language instruction for analysis.",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
):
    """
    Analyzes a MAF file using a CrewAI workflow.
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

        # Create the tasks
        parsing_task = Task(
            description=f"Parse the instruction: {instruction}",
            agent=chief_analyst_agent,
            expected_output="A detailed plan of action based on the instruction.",
            inputs={"instruction": instruction},
        )

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
            inputs={"maf_file_path": maf_file_path},
        )

        # Create the Report Agent
        report_agent = ReportAgent(
            role="Report Generator",
            goal="Generate a Markdown report summarizing the outputs of all tools.",
            llm=OpenAI(model_name="gpt-4o-mini", temperature=0.7),
            verbose=True,
        )

        # Create the Report Task
        report_task = Task(
            description="Generate a Markdown report summarizing the outputs of all tools.",
            agent=report_agent,
            expected_output="A Markdown-formatted report.",
            inputs={},  # Inputs will be dynamically populated later
        )

        # Create the Crew
        crew = Crew(
            agents=[chief_analyst_agent, report_agent],
            tasks=[
                parsing_task,
                summarization_task,
                somatic_interactions_task,
                drug_gene_interaction_task,
                report_task,
            ],
            verbose=verbose,
        )

        print("[bold green]Running the Crew...[/]")
        result = crew.kickoff()

        # Collect outputs from tasks
        tool_outputs = {
            "MAF Summarizer": result.tasks_output.get(
                summarization_task.description, "No output"
            ),
            "Somatic Interactions": result.tasks_output.get(
                somatic_interactions_task.description, "No output"
            ),
            "Drug-Gene Interactions": result.tasks_output.get(
                drug_gene_interaction_task.description, "No output"
            ),
        }

        # Generate the report
        report = report_agent._run(tool_outputs)
        print("[bold green]Generated Report:[/]")
        print(report)

        print("[bold green]MAF analysis completed successfully![/]")

    except Exception as e:
        print(f"[bold red]Error: {e}[/]")


if __name__ == "__main__":
    app()
