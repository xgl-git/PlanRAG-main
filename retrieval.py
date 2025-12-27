import jieba
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
llm = OpenAI(
    model="gpt-4o-mini",
    api_key="",
    api_base=''
)
Settings.llm = llm

def retrieval(query, retriever):
    def preprocess_text(text: str):
        tokens = [tok for tok in jieba.cut(text, cut_all=False) if tok.strip()]
        return " ".join(tokens)
    custom_prompt = PromptTemplate(
        "Answer the following question.\n"
        "Question: {query_str}\n"
        "Context:\n{context_str}\n"
        "Respond ONLY with a valid JSON object in this format:\n"
        "{\n"
        '  "answer": "<string>",\n'
        '  "confidence": <float between 0 and 1>\n'
        "}"
    )
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=custom_prompt,
    )
    retrival_docs= [bm25_node.text for bm25_node in retriever.retrieve(preprocess_text(query))]
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    response = query_engine.query(query)
    import re, json

    raw = response.response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
    return parsed, retrival_docs
