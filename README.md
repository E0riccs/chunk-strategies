# RAG 分片策略评估框架

[English](./doc/README_en.md) | 中文

此项目提供了一个在 RAG(Retrieval Augmented Generation) 环境中评估不同分片策略在不同文件类型下表现的框架。
该框架通过 LLM 的应用，自动化完成包含客观指标与主观指标的评价，并且易于扩展。

## 项目结构

```
chunk-strategies/
├── config/                  # 配置文件
│   ├── *.yaml               # 定义实验的关键参数
│   └── prompts.md           # 包含用于大语言模型(LLM)的提示词
│
├── data/                    # 测试数据文件（由用户创建/加载）
│
├── results/                 # 分块文本，评估报告
│
├── src/                     
│   ├── chunker.py           # 实现不同的文本分块算法
│   ├── experiment_runner.py # 负责实验执行
│   ├── file_handler.py      # 处理文件类型和测试数据的加载与管理
│   ├── llm_handler.py       # 处理与大语言模型的交互
│   ├── rag_handler.py       # 处理检索增强生成（RAG）流程
│   ├── rag_utils/           # RAG 工具函数
│   ├── eval_utils/          # 评估工具函数
│   ├── llm_utils/           # LLM 工具函数
│   └── utils/               # 通用工具函数
│
├── main.py                  # 运行实验的主脚本
├── requirements.txt         
└── README.md                
```

## 功能特性

1.  **可配置的文件类型**：通过 `config/file_types.yaml` 定义不同类别的文本文件（例如章节式长文、条目列表、简短纯文本），每种类型指向一个示例数据文件。
2.  **可配置的分块策略**：通过 `config/chunking_strategies.yaml` 定义多种分块方法（例如简单分割、递归字符分割），并设置具体参数（分块大小、重叠量）。
3.  **支持自定义 LLM 处理 API**：通过`config/llm_info.yaml` 配置并使用自定义的 llm模型/embedding模型/reranker模型 API。
4.  **引入LLM的主观+客观评估**：
    *   总处理时间（分块时间）。
    *   利用 LLM 生成标准 Question-Answer 对，自动化评估特定分片策略在实际 RAG 中的应用效果。
5.  **基于配置文件的灵活实验**：
    *   实验细节全部由YAML配置文件统一定义
6.  **模块解耦**：系统设计上清晰分离各模块，便于开发者二次扩展。
7.  **支持不同等级的外部数据文件**：
    *   原始文本：用户可以只提供原始文本，即可使用预定义的分片策略/用户实现的分片策略进行评估。
    *   分片结果：用户可以只提供原始文本和分片结果，无需额外实现分片策略，即可进行评估。

## 安装配置
1.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```

2.  **设置LLM API密钥（用于基于LLM的评估）**：
    你可以将其设置为环境变量：
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    或者在运行 `main.py` 时通过命令行参数传入（见下文）。
    如果未提供API密钥，将跳过LLM评估。


## 快速开始
**运行单个实验：**

通过 config 配置文件中定义的名称指定文件类型（`config/file_types.yaml`）和分块策略（`config/chunking_strategies.yaml`）。

```bash
python main.py --file_type chapter_text --strategy simple_chunk_100_overlap_10
```

*批量运行实验：**

你可以在 `config/experiments_to_run.yaml` 文件中定义多个实验，他们将批量执行。

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

然后运行：

```bash
python main.py --run_all_from_config config/experiments_to_run.yaml
```
或
```bash
python main.py
```
``main.py`` 的命令行选项可见其内定义。

## 进阶实验
所有自定义行为均需参照现有实现。
1.  **添加新文件类型**：
    *   将原始文本文件 `new_file_name.txt` 添加到 `data/` 目录（或任意路径）。
    *   在 `config/file_types.yaml` 中定义新文件类型：
        ```yaml
        file_types:
          # ... 现有类型 ...
          - name: my_new_document_type
            description: "你的新文档类型描述。"
            test_file: "data/new_file_name.txt" # 相对于项目根目录的路径
        ```

2.  **添加新分块策略**：
    *   如果是现有方法的变体（如 `simple_split` 或 `recursive_character_text_splitter` 不同参数），在 `config/chunking_strategies.yaml` 添加新条目：
        ```yaml
        chunking_strategies:
          # ... 现有策略 ...
          - name: my_custom_recursive_split
            method: recursive_character_text_splitter # 或 simple_split
            params:
              chunk_size: 500
              chunk_overlap: 50
        ```
    *   如果是全新分块算法，需要：
        1.  在 `src/chunker.py/SpliterFactory` 中实现新的分割方法（如 `_my_new_splitter_method`）。
        2.  更新 `src/chunker.py/SpliterFactory.create_spliter` 方法，根据新 `method` 名称调用新方法。
        3.  在 `config/chunking_strategies.yaml` 中使用新 `method` 名称和参数定义策略。

3.  **添加新 LLM 模型**：
    1. 在 `config/llm_info.yaml` 中定义新模型的 API 信息。
    2. 如果是新平台, 请根据新引入的模型用途：
      1. LLM基本用途：对话
        *   请在 `src/llm_utils/api_factory.py` 中引入新 API 平台；
        *   请根据新平台的 API 规范，在 `src/llm_utils/apis/YOUR_NEW_API_PLATFORM.py` 中实现 API 接口，并实现 ``send_message``  和 ``answer_from_json`` 方法，并遵循响应模型。
      2. RAG基本用途：Embedding
        *   请在 `src/rag_utils/api_factory.py` 中引入新 API 平台；
        *   请根据新平台的 API 规范，在 `src/rag_utils/apis/YOUR_NEW_API_PLATFORM.py` 中实现 API 接口，并实现 ``get_embedding`` 和 ``embed_result_from_json`` 方法，并遵循响应模型。

3.  **修改评估**：
    *  通过修改 ``src/eval_utils/evalutor.py`` 可以引入新的评价标准，无需修改配置文件``config/experiments_to_run.yaml``
    *  修改配置文件``config/experiments_to_run.yaml`` 产生的新评价标准必须在 ``src/eval_utils/evalutor.py`` 中实现。


## 结果

*   **Question-Answer**：每个待测试文件生成的 Question-Answer 对将保存为 `.txt` 文件，位于 `data/qa_paris` 目录.
*   **分块**：若使用预定义/用户实现的分片策略，每个实验生成的分块文本保存为 `.txt` 文件，位于 `results/chunks` 目录.
*   **实验结果**：包含所有实验指标的 CSV 文件将储存于 ``results/summary``。

## 其他技术信息
1. 使用 ChromaDB 作为向量数据库存储分块文本。

## TODO

* Reranker 暂未支持