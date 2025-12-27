import json
import itertools
import openai
from collections import defaultdict, deque
from FlagEmbedding import BGEM3FlagModel
client = openai.OpenAI(
    api_key='',
    base_url=''
)
with open('./prompt/prompt2.txt', 'r', encoding = 'utf-8') as f:
    prompt2 = f.read()
with open('./prompt/prompt3.txt', 'r', encoding = 'utf-8') as f:
    prompt3 = f.read()
with open('./prompt/prompt4.txt', 'r', encoding = 'utf-8') as f:
    prompt4 = f.read()
bgem3 = BGEM3FlagModel('./bge', use_fp16=True)
def estimate_depth(nodes, edges):
    graph = defaultdict(list)
    indegree = defaultdict(int)
    for e in edges:
        graph[e["from"]].append(e["to"])
        indegree[e["to"]] += 1
    roots = [n for n in nodes if indegree[n] == 0]
    max_depth = 0
    for root in roots:
        q = deque([(root, 1)])
        while q:
            node, depth = q.popleft()
            max_depth = max(max_depth, depth)
            for nxt in graph[node]:
                q.append((nxt, depth + 1))
    return max_depth

def semantic_simlarity(plan, query, prompt):

    prompt = prompt.replace("[[LogicalQueryTreee]]", str(plan))
    output = call_gpt4o(prompt)

    embeddings_1 = bgem3.encode(query)['dense_vecs']
    embeddings_2 = bgem3.encode(output)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
    return float(similarity)
def compute_cost(left_nodes, right_nodes, left_edges, right_edges, score):
    nodes = set(left_nodes + right_nodes)
    edges = left_edges + right_edges

    node_cost = len(nodes)
    edge_cost = len(edges) * 0.5
    depth_penalty = estimate_depth(nodes, edges) * 0.3
    semantic_penalty = score * 5
    balance_penalty = abs(len(left_nodes) - len(right_nodes)) * 0.2

    return node_cost + edge_cost - depth_penalty + semantic_penalty + balance_penalty
def has_cycle(edges):
    """
    检测是否存在环路
    edges: List[Dict], 每个元素形如 {"from": node1, "to": node2, "type": ...}
    """


    # 构建邻接表
    graph = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set()

    for e in edges:
        u, v = e["from"], e["to"]
        graph[u].append(v)
        indegree[v] += 1
        nodes.add(u)
        nodes.add(v)

    # 拓扑排序
    q = deque([n for n in nodes if indegree[n] == 0])
    visited = 0
    while q:
        node = q.popleft()
        visited += 1
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    return visited != len(nodes)
def call_gpt4o(prompt: str) -> str:
    request_params = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
            {"role": "user", "content": prompt},
        ],
    }
    response = client.chat.completions.create(**request_params)
    response = response.model_dump()
    return response['choices'][0]['message']['content'].strip()

def can_join(atom1: str, atom2: str, query: str, atomic_queries: dict, prompt: str, built_tree = None):
    prompt = prompt.replace("[[query]]", query)\
        .replace("[[atomic queries]]", str(atomic_queries))\
        .replace("[[atom1]]", atom1).replace("[[atom2]]", atom2).replace("[[built_tree]]", str(built_tree))
    output = call_gpt4o(prompt)
    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        result = {"relation_type": 0}
    return result


def query2lqt(atomic_queries, query):
    all_queries = list(atomic_queries.keys())
    pair_results = {}
    for l, r in itertools.combinations(all_queries, 2):
        pair_results[(l,r)] = can_join(
            atomic_queries[l]["question"],
            atomic_queries[r]["question"],
            query,
            atomic_queries,
            prompt2
        )
    def get_pair_result(l, r):
        key = tuple(sorted([l, r]))
        return pair_results.get(key, {"relation_type": 0})


    DP = {}
    for q in all_queries:
        DP[frozenset([q])] = {"cost": 1, "tree": {"nodes": [q], "edges": []}}
    edges = None
    for size in range(2, len(all_queries)+1):
        for subset in itertools.combinations(all_queries, size):
            subset = frozenset(subset)
            best_plan = None
            best_cost = float("-inf")
            for split_size in range(1, size):
                for left in itertools.combinations(subset, split_size):
                    left = frozenset(left)
                    right = subset - left
                    if left in DP and right in DP:
                        joinable = False
                        for l in left:
                            for r in right:
                                if get_pair_result(l, r)["relation_type"] == 0:
                                    continue
                                result = can_join(atomic_queries[l]["question"], atomic_queries[r]["question"], query,
                                                  atomic_queries, prompt2, best_plan)
                                relation_type = result["relation_type"]

                                if relation_type != 0:
                                    if relation_type == 1:

                                        direct = result.get("direct", 3)
                                        if direct == 3:
                                            join_edge = (r, l, relation_type)
                                        else:
                                            join_edge = (l, r, relation_type)
                                        edges = DP[left]["tree"]["edges"] + DP[right]["tree"]["edges"] + \
                                                ([{"from": join_edge[0], "to": join_edge[1],
                                                   "type": join_edge[2]}] if join_edge else [])
                                        if has_cycle(edges):
                                            break
                                    else:
                                        join_edge = (l, r, relation_type)
                                    joinable = True
                                    break
                            if joinable:
                                break
                        if not joinable:
                            continue
                        nodes = DP[left]["tree"]["nodes"] + DP[right]["tree"]["nodes"]
                        nodes1 = [{"id": nid, "query": atomic_queries[nid]["question"]} for nid in nodes]
                        edges = DP[left]["tree"]["edges"] + DP[right]["tree"]["edges"] + \
                                ([{"from": join_edge[0], "to": join_edge[1], "type": join_edge[2]}] if join_edge else [])
                        score = semantic_simlarity({"nodes": nodes1, "edges": edges}, query, prompt3)
                        cost = compute_cost(
                            DP[left]["tree"]["nodes"],
                            DP[right]["tree"]["nodes"],
                            DP[left]["tree"]["edges"],
                            DP[right]["tree"]["edges"],
                            score
                        )
                        if cost > best_cost:
                            best_cost = cost
                            best_plan = {"nodes": nodes, "edges": edges}
            if best_plan:
                DP[subset] = {"cost": best_cost, "tree": best_plan}
    full_set = frozenset(all_queries)
    if full_set not in DP:
        print("⚠️ Warning: cannot build full query tree, fallback to largest subset")
        full_set = max(DP.keys(), key=len)

    final_tree = DP[full_set]["tree"]
    nodes = []
    for nid in final_tree["nodes"]:
        info = atomic_queries[nid]
        node = {"id": nid, "type": info["type"], "query": info["question"]}
        nodes.append(node)

    logical_query_tree = {"nodes": nodes, "edges": final_tree["edges"]}
    print(json.dumps(logical_query_tree, indent=2, ensure_ascii=False))
