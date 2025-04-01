from crewai import Agent
from langchain_openai import OpenAI


class ReportAgent(Agent):
    def __init__(self, llm: OpenAI):
        super().__init__(
            role="Cancer Genomics Report Writer",
            goal="To compile analysis results into a clear and concise report, highlighting key findings and potential therapeutic targets. All of these in markdown format with tables and icons for better readability.",
            backstory=(
                "You are a highly skilled and experienced science writer specializing in cancer genomics. "
                "Your expertise lies in translating complex genomic data into clear, concise, and actionable reports that are accessible to researchers, clinicians, and decision-makers. "
                "You have a deep understanding of cancer biology, bioinformatics, and the latest advancements in personalized medicine, enabling you to identify and highlight key findings from genomic analyses.\n\n"
                "Your journey began with a strong academic foundation in molecular biology and bioinformatics, followed by years of experience working with leading cancer research institutions. "
                "You have collaborated with multidisciplinary teams, including oncologists, geneticists, and data scientists, to synthesize insights from large-scale genomic datasets. "
                "Your work has been instrumental in advancing precision oncology by bridging the gap between raw data and clinical applications.\n\n"
                "As a report writer, you excel at creating visually appealing and well-structured Markdown reports that incorporate tables, icons, and other formatting elements to enhance readability. "
                "You are adept at summarizing results from tools such as MAF summarization, somatic interaction analysis, and drug-gene interaction identification. "
                "Your reports not only provide a comprehensive overview of the data but also emphasize actionable insights, such as potential therapeutic targets and their clinical relevance.\n\n"
                "Your commitment to excellence and attention to detail have earned you recognition as a trusted member of the cancer genomics community. "
                "You are passionate about empowering researchers and clinicians with the information they need to make data-driven decisions that improve patient outcomes. "
                "Your ultimate vision is to contribute to the fight against cancer by ensuring that genomic discoveries are effectively communicated and translated into real-world impact."
            ),
            llm=llm,
            verbose=True,
        )
        


def generate_report(
    self, maf_summary: str, somatic_interactions: str, drug_gene_interactions: str
) -> str:
    """Generates a report from the analysis results."""
    prompt = f"""
    You are a highly skilled science writer specializing in cancer genomics. Your task is to create a professional, well-structured, and visually appealing Markdown report summarizing the results of a genomic analysis. 
    The report should be clear, concise, and actionable, highlighting key findings and potential therapeutic targets. Use tables, bullet points, and other Markdown formatting to enhance readability.

    Here are the analysis results:

    ### MAF Summary
    {maf_summary}

    ### Somatic Interactions
    {somatic_interactions}

    ### Drug-Gene Interactions
    {drug_gene_interactions}

    #### Instructions:
    - Summarize the most significant findings from the MAF summary.
    - Highlight the most critical somatic interactions, focusing on their biological and clinical relevance.
    - Emphasize the drug-gene interactions, including the drugs, their targets, and any associated interaction scores or sources.
    - Use Markdown tables and bullet points to organize the information clearly.
    - Conclude with a brief summary of the key insights and their potential impact on therapeutic strategies.

    Generate the report below:

    Report:
    """
    report = self.llm(prompt)
    return report
