# main.py
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


# Function to create the chief analyst agent
def create_chief_analyst(maf_summarizer, somatic_interactions, drug_gene_interactions):
    llm = OpenAI(temperature=0.7)
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
            tools=[
                NaturalLanguageParser(),  # Instantiate the tool
                TaskDelegator(),  # Instantiate the tool
                maf_summarizer,  # Already instantiated
                somatic_interactions,  # Already instantiated
                drug_gene_interactions,  # Already instantiated
            ],
            verbose=True,
        )
    except Exception as e:
        print(f"Error creating Chief Cancer Genomics Analyst: {e}")


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
        maf_summarizer_tool = MAFSummarizer()
        somatic_interactions_tool = SomaticInteractionsTool()
        drug_gene_interaction_tool = DrugGeneInteractionTool()

        # Create the chief analyst agent
        chief_analyst_agent = create_chief_analyst(
            maf_summarizer_tool, somatic_interactions_tool, drug_gene_interaction_tool
        )

        # Create the tasks
        parsing_task = Task(
            description=f"Parse the instruction: {instruction}",
            agent=chief_analyst_agent,
            expected_output="A detailed plan of action based on the instruction.",
            inputs={"instruction": instruction},  # Pass input as a dictionary
        )

        delegation_task = Task(
            description=f"Delegate the tasks based on the plan generated in the previous step. The MAF file is located at: {maf_file_path}",
            agent=chief_analyst_agent,
            expected_output="Tasks delegated to the appropriate tools.",
            inputs={
                "plan_json": '{"steps": ["Summarize MAF file", "Perform somatic interaction analysis", "Identify drug-gene interactions"]}',
                "maf_file_path": maf_file_path,
            },  # Pass input as a dictionary
        )

        summarization_task = Task(
            description=f"Summarize the MAF file located at: {maf_file_path}",
            agent=chief_analyst_agent,
            expected_output="A summary of the MAF file.",
            inputs={"maf_file_path": maf_file_path},  # Pass input as a dictionary
        )

        somatic_interactions_task = Task(
            description=f"Perform somatic interaction analysis on the MAF file located at: {maf_file_path}",
            agent=chief_analyst_agent,
            expected_output="Somatic interaction analysis results.",
            inputs={
                "maf_file_path": maf_file_path,
                "top_n": 25,
                "pvalue_cutoff": 0.05,
            },  # Pass input as a dictionary
        )

        drug_gene_interaction_task = Task(
            description=f"Identify potential therapeutic targets from the MAF file located at: {maf_file_path}",
            agent=chief_analyst_agent,
            expected_output="Potential therapeutic targets identified.",
            inputs={"maf_file_path": maf_file_path},  # Pass input as a dictionary
        )

        # Create the Crew
        crew = Crew(
            agents=[chief_analyst_agent],
            tasks=[
                parsing_task,
                delegation_task,
                summarization_task,
                somatic_interactions_task,
                drug_gene_interaction_task,
            ],
            verbose=2 if verbose else 0,
        )

        print("[bold green]Running the Crew...[/]")
        result = crew.kickoff()
        print("[bold green]Crew Result:[/]")
        print(result)

    except Exception as e:
        print(f"[bold red]Error: {e}[/]")


if __name__ == "__main__":
    app()
