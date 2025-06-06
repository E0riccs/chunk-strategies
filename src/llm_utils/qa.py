import json
import re

def trans_to_qa_list(llm_answer: str):
    """
    args:
        llm_answer: str, the answer from llm
    return:
        list: A list of QA pair dictionaries if successful, otherwise an empty list.
    Transforms the pure text llm answer to a list of qa pairs.
    Example:
    Input llm_answer:
    "```json\n[
        {
            \"question\": \"What is AI?",
            \"answers\": [
                \"Artificial Intelligence is...\",
                \"AI is the simulation of human intelligence...\"
            ]
        },
        {
            \"question\": \"What is machine learning?",
            \"answers\": [
                \"Machine learning is a subset of AI...\",
                \"It involves algorithms that learn from data...\"
            ]
        }
    ]\n```"
    Output:
    [
        {
            "question": "What is AI?",
            "answers": [
                "Artificial Intelligence is...",
                "AI is the simulation of human intelligence..."
            ]
        },
        {
            "question": "What is machine learning?",
            "answers": [
                "Machine learning is a subset of AI...",
                "It involves algorithms that learn from data..."
            ]
        }
    ]
    """
    if not isinstance(llm_answer, str):
        return []

    # try to find the first '[' or '{' and last ']' or '}'
    start_bracket = -1
    end_bracket = -1
    
    first_curly = llm_answer.find('{')
    first_square = llm_answer.find('[')
    
    if first_curly != -1 and (first_square == -1 or first_curly < first_square):
        start_bracket = first_curly
    elif first_square != -1:
        start_bracket = first_square
        
    last_curly = llm_answer.rfind('}')
    last_square = llm_answer.rfind(']')
    
    if last_curly != -1 and (last_square == -1 or last_curly > last_square):
        end_bracket = last_curly
    elif last_square != -1:
        end_bracket = last_square

    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        json_str = llm_answer[start_bracket : end_bracket+1]
    else:
        json_str = llm_answer

    # 解析
    try:
        # Clean up potential escape issues common in LLM outputs
        json_str = json_str.replace('\\"', '"') # Replace escaped quotes if LLM over-escapes
        json_str = json_str.replace('\\n', '').replace('\n', '') # Replace escaped newlines
        
        # Attempt to parse the extracted string
        data = json.loads(json_str)
        if isinstance(data, list):
            # Validate structure for QA pairs
            valid_pairs = []
            for item in data:
                if isinstance(item, dict) and 'question' in item and 'answers' in item and isinstance(item['answers'], list):
                    valid_pairs.append(item)
                else:
                    # If an item is not a valid QA pair, we might skip it or handle error
                    # For now, let's be strict and only include valid ones
                    pass 
            return valid_pairs
        elif isinstance(data, dict) and 'question' in data and 'answers' in data and isinstance(data['answers'], list):
            # Handle case where a single QA pair is returned as a dict instead of a list of one dict
            return [data]
        return [] # Return empty list if not a list of QA dicts or a single QA dict
    except json.JSONDecodeError:
        # Fallback: try to find individual JSON objects if the main parse fails
        # This is a more aggressive cleanup for very messy outputs.
        # It looks for patterns like {"question": ..., "answers": ...}
        qa_pairs = []
        try:
            # Regex to find JSON objects that look like QA pairs
            # This is a simplified regex and might need refinement for complex cases
            pattern = re.compile(r'\{\s*"question"\s*:\s*".*?"\s*,\s*"answers"\s*:\s*\[.*?\]\s*\}', re.DOTALL)
            for match_obj in pattern.finditer(llm_answer):
                try:
                    qa_pair = json.loads(match_obj.group(0))
                    if isinstance(qa_pair, dict) and 'question' in qa_pair and 'answers' in qa_pair and isinstance(qa_pair['answers'], list):
                        qa_pairs.append(qa_pair)
                except json.JSONDecodeError:
                    continue # Skip if a specific match fails to parse
            return qa_pairs
        except Exception:
            return [] # Final fallback
    except Exception:
        return [] # Catch any other unexpected errors

def extract_qa_pairs(llm_answer):
    """
    args:
        llm_answer: str, the answer from llm
    Extracts QA pairs from text llm response.
    """

    qa_list = trans_to_qa_list(llm_answer)

    qa_pairs = []
    for qa in qa_list:
        qa_pairs.append({
            'question': qa['question'],
            'answers': qa['answers']
        })
    return qa_pairs


if __name__ == '__main__':
    text_dict = '''
'\n```json\n[\n    {\n        "question": "根据《文物保护法》，我国不可移动文物的保护等级分为哪些级别？",\n        "answers": [\n            "标准答案：全国重点文物保护单位、省级文物保护单位、设区的市级/县级文物保护单位（根据第二章第23-25条）。",\n            "补充答案：全国重点文保单位需由国务院批准（第23条），省级由省政府核定并报国务院备案（第23条）。",\n            "补充答案：未定级不可动文物由县级政府登记备案并向社会公布（第23条）"\n        ]\n    },\n    {\n        "question": "国有可移动文物在什么情况下所有权会改变？",\n        "answers": [\n            "标准答案：国有可移动文物所有权不因收藏单位变更而改变（第六条）。",\n            "补充答案：若收藏单位终止运营且无合法继任者时需重新界定（隐含于第六条）。",\n            "补充答案：调拨至其他国有机构时所有权不变（第四章第五十四条）"\n        ]\n    },\n    {\n        "question": "发现地下文物后应如何处理？",\n        "answers": [\n            "标准答案：立即报告当地文物部门（第四十六条），24小时内到场并7日内提出处理意见（第四十六条）。",\n            "补充答案：涉及重要文物需同时上报国务院文物部门（第四十六条）。",\n            "补充案例：2020年某基建项目发现明代古城墙遗址后按此流程报批"\n        ]\n    },\n    {\n        "question": "非国有不可移动文物转让给外国人的法律后果是什么？",\n        "answers": [\n            "标准答案：禁止转让/抵押给外国人（第三十六条）。",\n            "补充说明：转让需经省级以上政府批准并报国务院备案（第三十条）。",\n            "案例延伸：2019年某私人收藏的清代匾额非法走私出境被追回"\n        ]\n    },\n    {\n        "question": "博物馆如何合规利用馆藏一级文物？",\n        "answers": [\n            "标准答案：修复/复制需国务院批准；展览需核实来源合法性（第六十二条、第五十五条）。",\n            "补充规定：调拨需省级政府批准并备案全国一级文物档案（第五十四条）。",\n            "监管措施：离任审计必须核查馆藏一级文物移交记录（第六十三条）"\n        ]\n    },\n    {\n        "question": 违反文物保护法造成严重损害'
'''
    qa_pairs = trans_to_qa_list(text_dict)
    print(qa_pairs)