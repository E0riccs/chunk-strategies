def extract_qa_pairs(json_response):
    """
    Extracts QA pairs from a JSON response.
    """
    qa_pairs = []
    for qa in json_response['choices'][0]['message']['content']:
        qa_pairs.append({
            'question': qa['question'],
            'answers': qa['answers']
        })
    return qa_pairs