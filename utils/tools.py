import torch
import random
from IPython.display import Markdown, display
import re

from pydantic import BaseModel, Field

def clear():
    torch.cuda.empty_cache()


def md(*args):
    content = "\n".join(str(arg) for arg in args)
    display(Markdown(content))
    

def blue(text):
    return f"<span style='color: blue; font-weight: bold;'>{text}</span>"


def grey(text):
    return f"<span style='color: grey; font-weight: bold;'>{text}</span>"


def green(text):
    return f"<span style='color: green; font-weight: bold;'>{text}</span>"

    
def get_vectordb_length(vectordb):
    collections = vectordb._client.list_collections()
    total_length = 0

    for collection in collections:
        total_length += collection.count()

    return total_length


def search(vectordb, text, k=5, method='MMR'):
    if method == 'MMR':
        return vectordb.max_marginal_relevance_search(text,k=k)
    elif method == "SIM":
        return vectordb.similarity_search(text,k=2*k)
    

# 定义相似度搜索为一个函数
def similarity_search(vectordb, input_text, k=5, metadata_filter=None):
    if metadata_filter:
        candidate_docs = vectordb.similarity_search(input_text, k=k*5)
    else:
        candidate_docs = vectordb.similarity_search(input_text, k=k)

    
    if metadata_filter:
        filtered_docs = [doc for doc in candidate_docs if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())]
    else:
        filtered_docs = candidate_docs
    
    return filtered_docs[:k]

def docl_search(vectordb, input_text, k=6):
    categories = ['not_hate', 'explicit_hate', 'implicit_hate']
    filtered_docs = []
    for category in categories:
        filtered_docs += similarity_search(vectordb, input_text, metadata_filter={"class": category}, k=k//len(categories))
    random.shuffle(filtered_docs)
    return filtered_docs


# 创建一个生成参考示例的函数
def build_reference_examples_chain(sim_docs):
    return "\n".join([f"Post: {doc.page_content}\nClass: {doc.metadata['class']}\n" for doc in sim_docs])


# Pydantic_Structured_result
class Analyse_Class(BaseModel):
    analyse: str = Field(description="analyse the post, examine weather it is hate speech or not")
    result: str = Field(description="(A)Hate or (B)Not Hate")
    

def extract_json(output):
    # 使用非贪婪匹配模式 (.*?) 和 re.DOTALL 仅匹配第一个 JSON 对象
    match = re.search(r'\{.*?\}', output, re.DOTALL)
    if match:
        try:
            return match.group()
        except json.JSONDecodeError:
            return "Error: Invalid JSON"
    else:
        return "Error: No JSON found"


if __name__ == "__main__":
    pass