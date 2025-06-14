import os

from src.llm_utils.api_factory import APIFactory
from src.llm_utils.qa import extract_qa_pairs
from src.llm_utils.response_model import LLMResponseModel

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

        qa_pairs = extract_qa_pairs(raw_ans[LLMResponseModel.ANS_CONTENT])

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
        """

        prompt = self._load_prompt(task)
        if not prompt:
            print(f"Warning: Prompt for task '{task}' not found. Using a generic approach.")
            prompt = " "

        user_content = self.build_content(prompt=prompt, **kwargs)
        
        raw_response = self._call_llm(user_content)
        response = self.api.answer_from_json(raw_response)[LLMResponseModel.ANS_CONTENT]

        return response


    def evaluate_rag_answer(self, question, generated_answer, reference_answer=None, eval_model_id='default_model'):
        """
        Evaluates a single RAG answer, potentially against a reference answer using an LLM.

        Args:
            question (str): The question that was asked.
            generated_answer (str): The answer generated by the RAG pipeline.
            reference_answer (str, optional): The ground truth or ideal answer.
            eval_model_id (str): The LLM model ID to use for evaluation.

        Returns:
            dict: Evaluation metrics (e.g., score, reasoning).
        """
        """
        计算QA对的召回率；/ 使用大模型（large）比较Q-A-A，给出主观评分；
        Args:
            generated_qa_pairs (list of dict): 原始QA对，每个元素含 'question', 'answer'.
            reranked_answers (list of dict): reranker回答的QA对，每个元素含 'question', 'reranked_answer'.
        Returns:
            dict: 包含评估结果，例如 'recall' 或 'subjective_scores'.
        """
        print(f"Evaluating RAG answer for question: '{question}'")
        eval_api = self.api # Default to the handler's main API
        if eval_model_id != self.model_id:
            try:
                eval_api_factory = APIFactory(model_id=eval_model_id, config_path=self.config_file_path)
                eval_api = eval_api_factory.create_api()
                print(f"Using LLM '{eval_model_id}' for evaluation.")
            except Exception as e:
                print(f"Warning: Could not load LLM '{eval_model_id}' for evaluation. Falling back to default. Error: {e}")
        
        # Define a JSON schema for the evaluation output
        # This helps in getting structured output from the LLM
        evaluation_schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "description": "Faithfulness and relevance score from 1 (poor) to 5 (excellent)."},
                "reason": {"type": "string", "description": "Brief explanation for the score."},
                "critique": {"type": "string", "description": "Suggestions for improvement, if any."}
            },
            "required": ["score", "reason"]
        }

        eval_prompt_template = self._load_prompt("RAGEval") # Expects RAGEval in prompts.md
        if not eval_prompt_template:
            print("Warning: RAGEval prompt not found. Using a default evaluation prompt.")
            if reference_answer:
                user_content = f"Question: {question}\nReference Answer: {reference_answer}\nGenerated Answer: {generated_answer}\n\nPlease evaluate the Generated Answer based on its faithfulness to the context (if provided implicitly) and relevance to the Question. If a Reference Answer is provided, also consider its correctness compared to it. Provide a score from 1 to 5 (5 is best) and a brief reason. Format your response as a JSON object with keys 'score' (integer) and 'reason' (string)."
            else:
                user_content = f"Question: {question}\nGenerated Answer: {generated_answer}\n\nPlease evaluate the Generated Answer based on its relevance to the Question and general quality. Provide a score from 1 to 5 (5 is best) and a brief reason. Format your response as a JSON object with keys 'score' (integer) and 'reason' (string)."
        else:
            user_content = eval_prompt_template.replace("{question}", question)\
                                            .replace("{generated_answer}", generated_answer)\
                                            .replace("{reference_answer}", reference_answer if reference_answer else "N/A")
        
        # Instruct the LLM to respond in JSON format according to the schema
        # This might require specific prompting techniques depending on the LLM API
        # For OpenAI, you can use the `response_format` parameter in newer API versions.
        # For others, you might add instructions like "Please respond in JSON format matching this schema: {json_schema_string}"
        # For simplicity, we'll assume the prompt guides the LLM sufficiently or the API handles JSON output.
        
        # Add JSON schema instruction to the user content if not using a specific API feature for JSON mode
        if not (hasattr(eval_api, 'supports_json_mode') and eval_api.supports_json_mode()):
             user_content += f"\n\nRespond with a JSON object matching the following schema: {evaluation_schema}"

        raw_eval_response = eval_api.send_message(user_content, response_format={'type': 'json_object'}) # Assuming send_message can take response_format
        
        # The answer_from_json might need to be robust to parse the JSON string if the API returns it as a string
        evaluation_result = eval_api.answer_from_json(raw_eval_response, is_direct_json=True) # is_direct_json if API returns parsed dict

        if not isinstance(evaluation_result, dict) or not all(k in evaluation_result for k in evaluation_schema['required']):
            print(f"Warning: LLM evaluation did not return the expected JSON structure. Response: {evaluation_result}")
            return {"score": "N/A", "reason": "Error in parsing LLM evaluation.", "critique": "N/A", "raw_response": evaluation_result}

        print(f"Evaluation - Score: {evaluation_result.get('score')}, Reason: {evaluation_result.get('reason')}")
        return evaluation_result

# 示例用法 (可以放在 main.py 或测试脚本中)
if __name__ == '__main__':
    # 假设这是你的原始文本
    sample_original_text = "这是用于测试LLM评估器的一段示例文本。它包含了一些基本信息，以便生成问答对并进行评估。"
    sample_chunks = [
        "这是用于测试LLM评估器的一段示例文本。",
        "它包含了一些基本信息，以便生成问答对并进行评估。",
        "评估的目的是验证分块策略和问答系统的有效性。"
    ]

    # 初始化LLMEvaler
    # 注意：确保 config/prompt.txt 文件存在，或者修改 _load_prompt 中的默认行为
    # 如果需要加载模型参数，请取消 _load_model_params 相关代码的注释，并创建对应的yaml文件
    evaler = LLMEvaler(config_path="../config", output_dir="../results/qa_pairs_test")

    # 1. 生成QA对
    # file_type 可以是文件名或者描述性字符串，用于区分不同的QA对文件
    generated_qa = evaler.generate_qa_pairs(sample_original_text, file_type="sample_text")
    # generated_qa 会是类似 [{'question': 'Q1', 'answer': 'A1'}, ...]

    if generated_qa:
        questions_for_reranker = [qa['question'] for qa in generated_qa]

        # 2. 使用reranker和small LLM回答问题
        # 这里的 sample_chunks 是示例分块结果
        reranked_q_and_a = evaler.answer_questions_with_reranker(sample_chunks, questions_for_reranker)
        # reranked_q_and_a 会是类似 [{'question': 'Q1', 'reranked_answer': 'Ans1'}, ...]

        # 3. 评估回答
        evaluation_results = evaler.evaluate_answers(generated_qa, reranked_q_and_a)
        print("\nEvaluation Results:")
        print(f"  Recall: {evaluation_results['recall']:.2f}")
        print("  Subjective Scores:")
        for score_info in evaluation_results['subjective_scores']:
            print(f"    Q: {score_info['question']}")
            print(f"       Original A: {score_info['original_answer']}")
            print(f"       Reranked A: {score_info['reranked_answer']}")
            print(f"       Score: {score_info['score']} ({score_info['reason']})")
            print("-" * 20)
    else:
        print("No QA pairs generated, skipping further steps.")

