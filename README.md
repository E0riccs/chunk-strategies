# RAG Chunking Strategy Evaluation Framework

This project provides a framework to test and evaluate different text chunking strategies for various file types.

## Project Structure

```
chunk-strategies/
├── config/                  # Configuration files
│   ├── file_types.yaml      # Defines different types of input files and their test data
│   ├── chunking_strategies.yaml # Defines various chunking methods and their parameters
├── data/                    # Test data files
│
├── results/                 # Output directory for chunked texts and evaluation reports
├── src/                     # Source code
│   ├── __init__.py
│   ├── utils.py             # Utility functions (e.g., YAML loading)
│   ├── file_handler.py      # Handles loading and managing file types and test data
│   ├── chunker.py           # Implements different text chunking algorithms
│   ├── evaluator.py         # Calculates evaluation metrics for chunking results
│   └── experiment_runner.py # Orchestrates the experiment execution
├── main.py                  # Main script to run experiments
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Features

1.  **Configurable File Types**: Define different categories of text documents (e.g., chaptered long-form, itemized lists, short plain text) via `config/file_types.yaml`. Each type points to a sample data file.
2.  **Configurable Chunking Strategies**: Define various chunking methods (e.g., simple splitting, recursive character splitting) with specific parameters (chunk size, overlap) via `config/chunking_strategies.yaml`.
3.  **Comprehensive Evaluation Metrics**:
    *   Total processing time (chunking time).
    *   LLM-based quality score (1-5 scale, requires OpenAI API key).
    *   Average cosine similarity between original text and generated chunks.
    *   Number of chunks generated.
4.  **Flexible Experiment Execution**:
    *   Run a single experiment for a specific file type and chunking strategy.
    *   Run a batch of experiments defined in a YAML configuration file.
5.  **Organized Output**:
    *   Each experiment saves its chunked text output to a separate file in the `results/` directory.
    *   All evaluation metrics are compiled into a summary CSV file in the `results/` directory.
6.  **Decoupled Modules**: The system is designed with clear separation of concerns for file handling, chunking, evaluation, and experiment orchestration, making it easier to extend.
7.  **LLM Integration**: Uses API for qualitative evaluation of chunks. (API key required).

## Setup
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up LLM API Key (for LLM-based evaluation)**:
    You can set it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Alternatively, you can pass it as a command-line argument when running `main.py` (see below).
    If no API key is provided, LLM evaluation will be skipped.

## Running Experiments

The main script to run experiments is `main.py`.

**Basic Usage (runs default experiments):**

If you run `main.py` without arguments, it will look for `config/experiments_to_run.yaml`. If this file doesn't exist, it will create a default one with a couple of example experiments and run them.

```bash
python main.py
```

**Running a Single Experiment:**

Specify the file type and chunking strategy by their names defined in the YAML configuration files.

```bash
python main.py --file_type chapter_text --strategy simple_chunk_100_overlap_0
```

Available file types and strategies can be found in `config/file_types.yaml` and `config/chunking_strategies.yaml` respectively.

**Running a Batch of Experiments from a Configuration File:**

You can define a list of experiments to run in a YAML file. For example, create `config/my_batch_experiments.yaml`:

```yaml
# config/my_batch_experiments.yaml
experiments:
  - file_type: chapter_text
    chunking_strategy: simple_chunk_100_overlap_0
  - file_type: chapter_text
    chunking_strategy: recursive_char_split_150_overlap_15
  - file_type: itemized_text
    chunking_strategy: simple_chunk_200_overlap_20
  - file_type: short_plain_text
    chunking_strategy: simple_chunk_100_overlap_0
```

Then run:

```bash
python main.py --run_all_from_config config/my_batch_experiments.yaml
```

**Command-line Options for `main.py`:**

*   `--file_type TEXT`: Name of the file type to process.
*   `--strategy TEXT`: Name of the chunking strategy to use.
*   `--run_all_from_config CONFIG_FILE_PATH`: Path to a YAML file defining batch experiments.
*   `--api_key TEXT`: Your OpenAI API key (overrides environment variable if set).
*   `--results_dir TEXT`: Directory to save results (default: `results/`).

## Customization

1.  **Adding New File Types**:
    *   Add your raw text file to the `data/` directory (or any path).
    *   Define the new file type in `config/file_types.yaml`:
        ```yaml
        file_types:
          # ... existing types ...
          - name: my_new_document_type
            description: "Description of your new document type."
            test_file: "data/my_new_file.txt" # Path relative to project root
        ```

2.  **Adding New Chunking Strategies**:
    *   If it's a variation of an existing method (e.g., `simple_split` or `recursive_character_text_splitter` with different parameters), add a new entry to `config/chunking_strategies.yaml`:
        ```yaml
        chunking_strategies:
          # ... existing strategies ...
          - name: my_custom_recursive_split
            method: recursive_character_text_splitter # Or simple_split
            params:
              chunk_size: 500
              chunk_overlap: 50
              # separators: ["\n# ", "\n## ", "\n\n"] # Optional for recursive
        ```
    *   If it's a completely new chunking algorithm, you'll need to:
        1.  Implement the new splitting logic as a method in `src/chunker.py` (e.g., `_my_new_splitter_method`).
        2.  Update the `Chunker.chunk()` method in `src/chunker.py` to call your new method based on a `method` name you define.
        3.  Define your new strategy in `config/chunking_strategies.yaml` using the new `method` name and any required `params`.

3.  **Modifying Evaluation**:
    *   The LLM prompt for evaluation is in `src/evaluator.py` within the `_get_llm_evaluation` method. You can tailor this prompt for more specific evaluation criteria.
    *   To add new metrics, modify `src/evaluator.py`.

## Output

*   **Chunked Text Files**: For each experiment, a `.txt` file containing the generated chunks will be saved in the `results/` directory. The filename will include the file type, strategy name, and a timestamp (e.g., `chapter_text_simple_chunk_100_overlap_0_20231027_123045123456_chunks.txt`).
*   **Summary CSV**: A CSV file (e.g., `experiment_summary_20231027_123500.csv`) will be created in `results/`, containing all metrics for every experiment run in a session. This allows for easy comparison across different configurations.

## TODO / Potential Enhancements

*   Add more sophisticated chunking strategies (e.g., semantic chunking, Markdown-aware splitting).
*   Implement more diverse evaluation metrics (e.g., chunk length distribution, overlap analysis).
*   Support for other LLM providers for evaluation.
*   More robust error handling and logging.
*   UI for easier configuration and result visualization.
*   Integration with experiment tracking tools (e.g., MLflow, Weights & Biases).