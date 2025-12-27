
from init_parameter import init_model
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from evaluate import *
import os
from query2lqt import *
from evaluate_sc import *
from iteration_retrieval_generation import *
parser = init_model()
args = parser.parse_args()

with open('./prompt/prompt1.txt', 'r', encoding='utf-8') as f:
    prompt1 = f.read()
with open('./prompt/prompt4.txt', 'r', encoding='utf-8') as f:
    prompt4 = f.read()
llm = OpenAI(
    model="gpt-4o-mini",
    api_key="",
    api_base='',
    temperature = 0.5
)
Settings.llm = llm
data_path = args.data_path
device = args.device
documents = [Document(text = json.loads(line)["text"].strip()) for line in open(os.path.join(data_path, 'corpus.jsonl'), 'r', encoding = 'utf-8').readlines()]
queries = [json.loads(line)['question'].strip() for line in open(os.path.join(data_path, 'queries.jsonl'), 'r', encoding = 'utf-8').readlines()]
gts1 = [[json.loads(line)['answer'].strip()] for line in open(os.path.join(data_path, 'queries.jsonl'), 'r', encoding = 'utf-8').readlines()]
gts2 = [json.loads(line)['answer'].strip() for line in open(os.path.join(data_path, 'queries.jsonl'), 'r', encoding = 'utf-8').readlines()]
'''
def retrival(query, retriever):
    custom_prompt = PromptTemplate(
        "You are a strict JSON-only answering assistant.\n"
        "You must output only valid JSON.\n"
        "If the context does not contain enough information to answer, output an empty string as the answer.\n"
        "Do NOT explain, do NOT say you cannot answer, do NOT output extra words.\n\n"
        "Question:\n{query_str}\n\n"
        "Context:\n{context_str}\n\n"
        "Output a single JSON object exactly in this format:\n"
        "{\n"
        '  "answer": "<string>"\n'
        "}"
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",  # 或 "tree_summarize"
        text_qa_template=custom_prompt,
    )
    retrival_docs = [node.text for node in retriever.retrieve(preprocess_text(query))]
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    response = query_engine.query(query).response
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if match:
        answer = json.loads(match.group())
    return answer, retrival_docs

'''

def preprocess_text(text: str):
    tokens = [tok for tok in jieba.cut(text, cut_all=False) if tok.strip()]
    return " ".join(tokens)

if args.retriever == "bm25":
    splitter = SentenceSplitter(chunk_size=10000)
    nodes = splitter.get_nodes_from_documents(documents)
    for node in nodes:
        node.metadata["original_text"] = node.text
        node.text = preprocess_text(node.text)
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=100)


def query2atomic(query, args, prompt):
    client = openai.OpenAI(
        api_key=args.api_key,
        base_url=''
    )

    prompt = prompt.replace("[[Problem]]", query)
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
    atomic_query = response['choices'][0]['message']['content'].strip()
    return atomic_query
preds = []
for query in queries:
    atomic_queries = query2atomic(query, args)
    lqt = query2lqt(atomic_queries, query)

    answer = iter_ret_gen(query, lqt, retriever)
    preds.append(answer)

result = evaluate_batch(preds, gts1, topk_list=[1, 3])
for k, v in result.items():
    print(f"{k}: {v:.2f}%")
acc_score = semantic_accuracy(preds, gts2)
print(f"Semantic Accuracy (Acc†): {acc_score:.2f}%")