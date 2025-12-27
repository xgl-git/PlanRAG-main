from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import re
import json
from retrieval import *
from llama_index.llms.modelscope import ModelScopeLLM
llm1 = ModelScopeLLM(model_name="LLM-Research/Meta-Llama-3-8B", model_revision="master", device_map="")
def iter_ret_gen(query, lqt, prompt, retriever):
    nodes = {node["id"]: node for node in lqt["nodes"]}
    children = defaultdict(list)
    parents = defaultdict(list)
    indegree = defaultdict(int)
    histoy = {node["id"]: [] for node in lqt["nodes"]}

    for edge in lqt["edges"]:
        u, v = edge["from"], edge["to"]
        children[u].append(v)
        parents[v].append(u)
        indegree[v] += 1

    ready = [nid for nid in nodes if indegree[nid] == 0]
    executed_order = []

    lock = threading.Lock()

    def execute_node(node, prompt, query):
        automic_query = node['query']

        previous_his = ""
        if parents[node["id"]]:
            his_list = [histoy[parent_id] for parent_id in parents[node["id"]] if histoy[parent_id]]
            for h in his_list:
                if len(h) == 3:
                    q, ans, docs = h
                    previous_his += f"previous question: {q}\nprevious answer: {ans}\n\n"

            prompt_filled = prompt.replace("[[Query]]", query)\
                                  .replace("[[automic_query]]", automic_query)\
                                  .replace("[[previous_his]]", previous_his)

            try:
                response = llm1.complete(prompt_filled)
                match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    automic_query = parsed.get('answer', automic_query)
                    print(automic_query)
            except Exception as e:
                print(f"LLM call failed: {e}")

        answer, docs = retrieval(automic_query, retriever)
        with lock:
            histoy[node["id"]] = [query, answer['answer'], docs]

        time.sleep(0.6)
        return node["id"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        while ready:
            futures = {executor.submit(execute_node, nodes[nid], prompt, query): nid for nid in ready}
            for future in as_completed(futures):
                nid = futures[future]
                executed_order.append(nid)

            new_ready = []
            for nid in ready:
                for v in children[nid]:
                    indegree[v] -= 1
                    if indegree[v] == 0:
                        new_ready.append(v)
            ready = new_ready

    return None