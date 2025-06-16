import os

from src.llm_utils.api_factory import APIFactory
from src.llm_utils.qa import extract_qa_pairs

from src.llm_utils.llm_model import LLMResponseModel, PromptKeywordsModel

class LLM_handler:
    def __init__(self, model_id, config_path="config"):
        self.model_id = model_id
        self.config_file_path = config_path

        self.load_api()

    def load_api(self):
        self.llm_api_factory = APIFactory(model_id=self.model_id, config_path=self.config_file_path)
        self.api = self.llm_api_factory.create_api()

    def _load_prompt(self, task):
        '''
        加载原始 Prompt

        args:
            task (str): 任务名称, 包括 RAGAnswer, GenQAs ...
        '''
        prompt_file_path = os.path.join(self.config_file_path, "prompts.md")

        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                # 根据任务，从不同的一级标题下内容下读取提示词
                prompt = ""
                for line in f:
                    if line.startswith("##"):
                        if line.strip() == f"## {task}":
                            # 读取下一行，直到遇到新的标题
                            next_line = f.readline()
                            while not next_line.startswith("##"):
                                prompt += next_line
                                next_line = f.readline()
                            break
                return prompt
                
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {prompt_file_path}")
        except Exception as e:
            print(f"Error loading prompt file: {e}")

    def build_content(self, prompt, original_text = '', **kwargs):
        '''
        根据原始提示词和附加信息，经过替换等操作得到最终的用户输入内容。
        替换形式：{key_word}
        
        args:
            prompt (str): 原始提示词
            original_text (str): 原始文本, 直接添加到后部
            **kwargs: 附加信息，变量名应该与提示词中的 {key_word} 一致
                     如果value是list类型，会自动转换为换行分隔的字符串
        '''
        for key, value in kwargs.items():
            # 如果value是list类型，转换为换行分隔的字符串
            if isinstance(value, list):
                value = '\n'.join(str(item) for item in value)
            prompt = prompt.replace(f"{key}", str(value))

        return prompt + '/n/n' + original_text

    def _call_llm(self, user_content):
        '''
        调用 LLM 的接口，以json格式返回回答。
        '''
        response = self.api.send_message(user_content)
        return response

    def _generate_qa_pairs(self, original_text, output_file_path, build_strategy='skip'):
        """
        使用大模型（large）针对原文生成QA对，并储存至本地 filetype.txt 文件中。
        
        该函数首先加载 GenQAs 任务的 prompt，接着使用大模型（large）来生成QA对。
        生成的QA对将以txt格式写入到 output_file_path 中。
        
        Args:
            original_text (str): 原始文本内容。
            output_file_path (str): 输出文件路径。
        """
        print(f"Generating QA pairs for {output_file_path}...")
        # 这里需要调用大模型（large）来生成QA对
        # 假设 call_large_llm 是一个调用大模型的函数
        
        # json_schema = {
        #     "score": "integer (0-100)",
        #     "reason": "string (评分理由)",
        #     "keywords": "array (提取的关键词)"
        # }
        prompt = self._load_prompt("GenQAs")
        raw_ans = self._call_llm(self.build_content(prompt=prompt, original_text=original_text)) # 传递加载的prompt和原文
        raw_ans = self.api.answer_from_json(raw_ans)

        # 或者使用新的 Pydantic 模型字段名（推荐）
        qa_pairs = extract_qa_pairs(raw_ans.ans_content)

        try:
            if not qa_pairs:
                print(f"No QA pairs generated for {output_file_path}.")
                return []
            
            # 追加策略
            if build_strategy == 'append':
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                existing_qa_pairs = extract_qa_pairs(existing_content)
                qa_pairs.extend(existing_qa_pairs)
                print(f"Appending QA pairs to {output_file_path}.")

            with open(output_file_path, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    f.write(f"Q: {qa['question']}\n")
                    f.write(f"A: {qa['answers']}\n\n")
            print(f"QA pairs saved to {output_file_path}")
        except Exception as e:
            print(f"Error writing QA pairs to file {output_file_path}: {e}")
        return qa_pairs

    def generate_qa_pairs(self, input_path='data', output_path='data/qa_pairs', build_strategy='skip'):
        """
        使用大模型（large）针对所有原文生成QA对。
        Args:
            input_path (str): 包含原始文本文件的目录路径。
            output_path (str): 保存生成的QA对文件的目录路径。
            build_strategy (str): 生成QA对的策略，'rebuild'/'append'/'skip'。
        """
        os.makedirs(output_path, exist_ok=True)
        print(f"Generating QA pairs from files in {input_path} to {output_path}")

        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path):
                file_name = os.path.splitext(filename)[0] # 从名字中获取文件类型
                output_file_path = os.path.join(output_path, f"{file_name}_qa_pairs.txt")

                if os.path.exists(output_file_path) and build_strategy == 'skip':
                    print(f"QA pairs for {file_name} already exist at {output_file_path}. Skipping.")
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_text = f.read()
                        self._generate_qa_pairs(original_text, output_file_path, build_strategy)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
        
    def chat(self, input_text= '', task='Normal', **kwargs):
        """
        Answers a single question using a RAG pipeline: retrieve, rerank, then generate answer.

        Args:
            input_text (str): The input text to chat.
            task (str): The task (RAGAnswer, GenQAs, etc.) of this chat.
            **kwargs: Additional keyword arguments to pass to the build_content method.
                      These will be validated using the PromptKeywords model.
        """

        prompt = self._load_prompt(task)
        if not prompt:
            print(f"Warning: Prompt for task '{task}' not found. Using a generic approach.")
            prompt = " "

        # Validate kwargs using Pydantic model
        try:
            user_content = self.build_content(prompt=prompt, original_text=input_text, **kwargs)
        except Exception as e:
            print(f"Warning: Invalid prompt keywords: {e}. Using raw kwargs.")
        
        raw_response = self._call_llm(user_content)
        response = self.api.answer_from_json(raw_response).ans_content

        return response

