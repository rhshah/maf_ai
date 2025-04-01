# MAF Analysis AI Workflow

This repository provides an AI-powered workflow for analyzing Mutation Annotation Format (MAF) files. It uses the CrewAI framework to orchestrate tasks such as summarizing MAF files, performing somatic interaction analysis, identifying drug-gene interactions, and generating a comprehensive Markdown report.

## Features

- **MAF Summarization**: Summarizes the key details of the MAF file.
- **Somatic Interaction Analysis**: Identifies significant somatic interactions.
- **Drug-Gene Interaction Identification**: Finds potential therapeutic targets.
- **Comprehensive Report Generation**: Combines all outputs into a Markdown report with tables and icons.

## Requirements

- Python 3.8 or higher
- Required Python packages (see `pyproject.toml` for details)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/maf_ai.git
   cd maf_ai
   ```
2. Install `pip-tools` and uv `globally` if not already installed:
   ```bash
   pip install pip-tools uv
   ```
3. Create a virtual environment and activate it:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
4. Install dependencies using uv:
   ```bash
   uv install
   ```
   This will automatically resolve and install all dependencies specified in pyproject.toml
5. Set up the environment variables:
   1. Create a .env file in the root directory.
   2. Add your OpenAI API key:

   ```bash
    OPENAI_API_KEY=your_openai_api_key
   ```
   Replace `your_api_key` with your actual CrewAI API key.

## Usage

Run the main.py script to analyze a MAF file and generate a Markdown report:
```bash
python main.py --maf_path path/to/your/maf_file.maf
```
Replace `path/to/your/maf_file.maf` with the actual path to your MAF file.

### Command Line Arguments

- `--maf-file-path`: Path to the MAF file to analyze.
- `--instruction`: Natural language instruction for the analysis.
- `--verbose`: Enable verbose output.
- `--output-file`: Path to save the generated Markdown report (default: maf_analysis_report.md).

### Example
```bash
python main.py --maf-file-path example.maf --instruction "Analyze the MAF file and summarize the key findings." --verbose
```
This will analyze the specified MAF file, summarize its contents, and generate a Markdown report with the findings.

## Output

The script generates a Markdown report summarizing the analysis results. Example sections include:

- MAF Summary
- Somatic Interactions
- Drug-Gene Interactions
- Conclusion

The report is saved to the specified output file.

## Example Output

The generated Markdown report includes:
- **MAF Summary**: A summary of the MAF file, including the number of mutations, genes involved, and other relevant statistics.
```bash
Number of Samples: 10129
Number of Genes: 414
Variant Classifications: {'Missense_Mutation': 55556, 'Nonsense_Mutation': 7936, ...}
```
- **Somatic Interactions**: A table of significant somatic interactions, including gene pairs and interaction scores.
```bash
| Gene A | Gene B | Interaction Score |
|--------|--------|------------------|
| TP53   | KRAS   | 0.85             |
| EGFR   | PIK3CA | 0.78             |
```
- **Drug-Gene Interactions**: A table of potential drug-gene interactions, including drug names and associated genes.
```bash
| Drug Name | Gene Name | Interaction Type |
|-----------|-----------|------------------|
| DrugA     | TP53      | Inhibitor        |
| DrugB     | KRAS      | Agonist          |
```
- **Conclusion**: This report summarizes the results of the MAF analysis, including the MAF file summary, somatic interaction analysis, and drug-gene interactions. The findings provide valuable insights into potential therapeutic targets and their clinical relevance..

### MAF Summary

## Project Structure
```bash
maf_ai/
├── [main.py]()                     # Entry point for the workflow
├── maf_tools/                  # Tools for MAF analysis
│   ├── maf_summarizer.py       # Summarizes MAF files
│   ├── somatic_interactions.py # Performs somatic interaction analysis
│   ├── drug_gene_interactions.py # Identifies drug-gene interactions
│   ├── natural_language_parser.py # Parses natural language instructions
│   ├── task_delegator.py       # Delegates tasks to agents
├── requirements.txt            # Python dependencies (auto-generated by pip-tools)
├── [pyproject.toml]()              # Dependency management configuration
├── .env                        # Environment variables (not included in repo)
└── [README.md]()                   # Project documentation
```     

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the AGPL License. See the `LICENSE` file for details.
