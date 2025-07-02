# RAG Chunking Strategy Evaluation Framework

English | [中文](../README.md)
This language version is translated by llm and checked by human.

This project provides a framework for evaluating the performance of different chunking strategies on various file types in a Retrieval Augmented Generation (RAG) environment.
The framework automates the evaluation of both objective and subjective metrics through the application of LLM and is designed for easy extension.

## Project Structure

```
chunk-strategies/
├── config/                  # Configuration files
│   ├── *.yaml               # Defines the key parameters for experiments
│   └── prompts.md           # Contains prompts for the Large Language Model (LLM)
│
├── data/                    # Test data files (Created/Loaded by yourself)
│
├── results/                 # Chunked text, evaluation reports
│
├── src/                     
│   ├── chunker.py           # Implements different text chunking algorithms
│   ├── experiment_runner.py # Orchestrates the experiment execution
│   ├── file_handler.py      # Handles loading and managing file types and test data
│   ├── llm_handler.py       # Handles interactions with the Large Language Model
│   ├── rag_handler.py       # Handles RAG (Retrieval-Augmented Generation) processes
│   ├── rag_utils/           # RAG utility functions
│   ├── eval_utils/          # Evaluation utility functions
│   ├── llm_utils/           # LLM utility functions
│   └── utils/               # General utility functions
│
├── main.py                  # Main script to run experiments
├── requirements.txt         
└── README.md                
```

## Features

1.  **Configurable File Types**: Define different categories of text documents (e.g., chaptered long-form, itemized lists, short plain text) via `config/file_types.yaml`. Each type points to a sample data file.
2.  **Configurable Chunking Strategies**: Define various chunking methods (e.g., simple splitting, recursive character splitting) with specific parameters (chunk size, overlap, etc.) via `config/chunking_strategies.yaml`.
3.  **Support for Custom LLM Processing APIs**: Configure and use custom LLM/embedding/reranker model APIs via `config/llm_info.yaml`.
4.  **Introduction of Subjective + Objective Evaluation using LLM**:
    *   Total processing time (chunking time).
    *   Utilize LLM to generate standard Question-Answer pairs to automate the evaluation of specific chunking strategies in a real RAG application.
5.  **Flexible Experiments Based on Configuration Files**:
    *   All experiment details are uniformly defined by YAML configuration files.
6.  **Decoupled Modules**: The system is designed with clear separation of concerns for each module, making it easier for developers to extend.
7.  **Support for Different Levels of External Data Files**:
    *   Raw Text: Users can provide only the raw text to evaluate with predefined or user-implemented chunking strategies.
    *   Chunked Results: Users can provide the raw text and the chunked results to perform evaluation without needing to implement the chunking strategy.

## Setup
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start
**Running a Single Experiment:**

Specify the file type (`config/file_types.yaml`) and chunking strategy (`config/chunking_strategies.yaml`) by their names defined in the config files.

```bash
python main.py --file_type chapter_text --strategy simple_chunk_100_overlap_10
```

**Running a Batch of Experiments:**

You can define multiple experiments in the `config/experiments_to_run.yaml` file, and they will be executed in a batch.

```yaml
experiments:
- file_type: rules_simple
  chunking_strategy: simple_chunk_1
  rerank: False
# - file_type: rules_simple
#   chunking_strategy: simple_chunk_1
#   rerank: True
#   reranker_method: rerank-english-v2.0
```

Then run:

```bash
python main.py --run_all_from_config config/experiments_to_run.yaml
```
or
```bash
python main.py
```
Command-line options for `main.py` can be found within the script itself.

## Advanced Experiments
All custom behaviors should refer to existing implementations.
1.  **Adding New File Types**:
    *   Add your raw text file `new_file_name.txt` to the `data/` directory (or any path).
    *   Define the new file type in `config/file_types.yaml`:
        ```yaml
        file_types:
          # ... existing types ...
          - name: my_new_document_type
            description: "Description of your new document type."
            test_file: "data/new_file_name.txt" # Path relative to project root
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
        ```
    *   If it's a completely new chunking algorithm, you'll need to:
        1.  Implement the new splitting logic in `src/chunker.py/SpliterFactory` (e.g., `_my_new_splitter_method`).
        2.  Update the `src/chunker.py/SpliterFactory.create_spliter` method to call your new method based on the new `method` name.
        3.  Define your new strategy in `config/chunking_strategies.yaml` using the new `method` name and any required `params`.

3.  **Adding New LLM Models**:
    1. Define the API information for the new model in `config/llm_info.yaml`.
    2. If it's a new platform, depending on the intended use of the model:
        1. Basic LLM use (dialogue):
            *   Import the new API platform in `src/llm_utils/api_factory.py`.
            *   Implement the API interface in `src/llm_utils/apis/YOUR_NEW_API_PLATFORM.py` according to the new platform's API specification, implementing communication method: `send_message`, and follow the response model to implement the information extraction method: `answer_from_json`.
        2. Basic RAG use (Embedding):
            *   Import the new API platform in `src/rag_utils/api_factory.py`.
            *   Implement the API interface in `src/rag_utils/apis/YOUR_NEW_API_PLATFORM.py` according to the new platform's API specification, implementing communication method: `get_embedding`, and follow the response model to implement the information extraction method: `embed_result_from_json`.

3.  **Modifying Evaluation**:
    *  You can introduce new evaluation criteria by modifying `src/eval_utils/evalutor.py` without changing the `config/experiments_to_run.yaml` file.
    *  New evaluation criteria generated by modifying `config/experiments_to_run.yaml` must be implemented in `src/eval_utils/evalutor.py`.


## Results

*   **Question-Answer**: The Question-Answer pairs generated for each test file will be saved as `.txt` files in the `data/qa_paris` directory.
*   **Chunks**: If using predefined/user-implemented chunking strategies, the generated text chunks for each experiment will be saved as `.txt` files in the `results/chunks` directory.
*   **Experiment Results**: A CSV file containing all experiment metrics will be stored in `results/summary`.

## Other Technical Information
1. Uses ChromaDB as the vector database to store chunked text.
