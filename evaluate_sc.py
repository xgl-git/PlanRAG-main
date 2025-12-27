
from typing import List



def semantic_accuracy(predictions: List[str], ground_truths: List[str], questions: List[str]) -> float:
    import openai

    client = openai.OpenAI(api_key="",
                           base_url='')
    assert len(predictions) == len(ground_truths) == len(questions)

    prompt_template = """In the following task, you are given a Question, 
a model Prediction for the Question, and a Ground-truth Answer to the Question. 
You should decide whether the model Prediction implies the Ground-truth Answer.

Question
{question}

Prediction
{prediction}

Ground-truth Answer
{answer}

Does the Prediction imply the Ground-truth Answer? Output Yes or No:
"""
    correct_count = 0
    for q, p, a in zip(questions, predictions, ground_truths):
        prompt = prompt_template.format(question=q, prediction=p, answer=a)
        request_params = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
                {"role": "user", "content": prompt},
            ],
        }
        response = client.chat.completions.create(**request_params)
        response = response.model_dump()
        response = response['choices'][0]['message']['content'].strip().lower()
        if "yes" in response:
            correct_count += 1
    return 100.0 * correct_count / len(predictions)