
from init_parameter import init_model
from llama_index.core import Document
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