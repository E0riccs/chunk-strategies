import os

from src.llm_utils.api_factory import APIFactory
from src.llm_utils.qa import extract_qa_pairs
from src.llm_utils.response_model import LLMResponseModel

# from sentence_transformers import CrossEncoder

class LLM_handler:
    def __init__(self, model_id, config_path="config"):
        self.model_id = model_id
        self.config_path = config_path
        self.config_file_path = os.path.join(self.config_path, "llm_info.yaml")

        self.load_api()

    def load_api(self):
        self.llm_api_factory = APIFactory(model_id=self.model_id, config_path=self.config_file_path)
        self.api = self.llm_api_factory.create_api()

    def _load_prompt(self, task):
        '''
        加载原始 Prompt
        '''
        prompt_file_path = os.path.join(self.config_path, "prompts.md")

        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                # 根据任务，从不同的一级标题下内容下读取提示词
                prompt = ""
                for line in f:
                    if line.startswith("##"):
                        if line.strip() == f"## {task}":
                            # 读取下一行，直到遇到新的一级标题
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

    def build_content(self, prompt, original_text, **kwargs):
        '''
        根据原始提示词和附加信息构建最终的用户输入内容
        '''
        return prompt + "\n\n" + original_text

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
        raw_ans = self._call_llm(self.build_content(prompt, original_text)) # 传递加载的prompt和原文
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
        

    def answer_questions_with_reranker(self, chunks, questions):
        """
        使用大模型（small）对分片结果，进行reranker后，回答Qs；
        Args:
            chunks (list of str): 文本分片列表。
            questions (list of str): 问题列表。
        Returns:
            list of dict: 每个字典包含 'question' 和 'reranked_answer'。
        """
        print("Answering questions with reranker...")
        results = []
        for q_idx, question_text in enumerate(questions):
            # 1. Rerank chunks for the current question (这里需要一个reranker实现)
            # 假设 rerank_chunks 是一个 reranker 函数
            # reranked_chunks = self.reranker.rerank(question_text, chunks)
            # For now, let's assume the first chunk is the most relevant after reranking
            if not chunks:
                print(f"Warning: No chunks provided for question: {question_text}")
                results.append({"question": question_text, "reranked_answer": "No chunks to process."})
                continue
            
            relevant_chunk = chunks[0] # 简化处理，实际应为rerank结果
            
            # 2. Use small LLM to answer the question based on the reranked chunk(s)
            # 假设 call_small_llm 是调用小模型的函数
            answer = self._call_llm(self.small_model, self.prompt, relevant_chunk, question_text)
            results.append({"question": question_text, "reranked_answer": answer})
            print(f"Q: {question_text} -> A (small_llm): {answer}")
        return results

    def evaluate_answers(self, generated_qa_pairs, reranked_answers):
        """
        计算QA对的召回率；/ 使用大模型（large）比较Q-A-A，给出主观评分；
        Args:
            generated_qa_pairs (list of dict): 原始QA对，每个元素含 'question', 'answer'.
            reranked_answers (list of dict): reranker回答的QA对，每个元素含 'question', 'reranked_answer'.
        Returns:
            dict: 包含评估结果，例如 'recall' 或 'subjective_scores'.
        """
        print("Evaluating answers...")
        # 选项1: 计算召回率 (简单示例，基于问题匹配)
        # 假设问题完全一致才算匹配
        # 注意：这是一个非常简化的召回率计算，实际可能需要更复杂的匹配逻辑（如语义相似度）
        
        # 创建一个从问题到原始答案的映射，方便查找
        original_answers_map = {qa['question']: qa['answer'] for qa in generated_qa_pairs}
        
        matched_questions = 0
        total_generated_questions = len(generated_qa_pairs)
        subjective_scores = []

        for reranked_qa in reranked_answers:
            question = reranked_qa['question']
            reranked_answer = reranked_qa['reranked_answer']
            
            if question in original_answers_map:
                matched_questions += 1
                original_answer = original_answers_map[question]
                # 选项2: 使用大模型（large）比较Q-A-A，给出主观评分
                score_prompt = f"原始问题: {question}\n原始答案: {original_answer}\n模型回答: {reranked_answer}\n请对模型回答的质量进行评分（1-5分，5分最好），并简要说明理由。"
                subjective_score_response = self._call_llm(self.large_model, score_prompt) 
                subjective_scores.append({
                    "question": question,
                    "original_answer": original_answer,
                    "reranked_answer": reranked_answer,
                    "score": subjective_score_response.get('score', 'N/A'),
                    "reason": subjective_score_response.get('reason', 'N/A')
                })
                print(f"Subjective score for Q: {question} -> Score: {subjective_score_response.get('score', 'N/A')}")
            else:
                 subjective_scores.append({
                    "question": question,
                    "original_answer": "N/A (Question not found in generated QA)",
                    "reranked_answer": reranked_answer,
                    "score": "N/A",
                    "reason": "原始QA中未找到此问题"
                })

        recall = (matched_questions / total_generated_questions) if total_generated_questions > 0 else 0
        print(f"Recall: {matched_questions}/{total_generated_questions} = {recall:.2f}")
        
        return {"recall": recall, "subjective_scores": subjective_scores}

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

