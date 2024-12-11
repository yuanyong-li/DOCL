from tqdm import tqdm
import torch
from langchain.embeddings.base import Embeddings


class BGEM3Embeddings(Embeddings):
    def __init__(self, model, tokenizer, batch_size=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.batch_size = batch_size  # 设置批处理大小

    def embed_text(self, text: str) -> list:
        # 单条文本的嵌入函数
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

    def embed_documents(self, texts: list) -> list:
        # 批量嵌入函数
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i+self.batch_size]
            # 对批量文本进行标记和张量化，并转移到设备
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 计算批次的平均嵌入并转换为列表
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> list:
        return self.embed_text(text)
    

class BERTEmbeddings:
    def __init__(self, model, tokenizer, max_length=512, batch_size=32):
        # 将模型和分词器设为实例变量
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device  # 获取模型的设备
        self.batch_size = batch_size  # 设置批处理大小
        self.max_length = max_length

    def embed_text(self, text: str) -> list:
        # 单条文本的嵌入函数
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
        return embeddings

    def embed_documents(self, texts: list) -> list:
        # 批量嵌入函数
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding documents"):
            batch_texts = texts[i:i+self.batch_size]
            # 对批量文本进行标记和张量化，并转移到设备
            inputs = self.tokenizer(batch_texts, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 计算批次的平均嵌入并转换为列表
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> list:
        return self.embed_text(text)


if __name__ == '__main__':
    from transformers import BertModel, BertTokenizer

    from modelscope import snapshot_download, AutoModel, AutoTokenizer
    import os
    # 加载 BERT 模型和分词器
    model = BertModel.from_pretrained("/home/liyuanyong2022/abusive/models/BERT/pretrained_models/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("/home/liyuanyong2022/abusive/models/BERT/pretrained_models/bert-base-uncased")

    # 将模型移动到可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化 BERTEmbeddings 并传入模型和分词器
    embedder = BERTEmbeddings(model=model, tokenizer=tokenizer, batch_size=16)

    # 单条文本嵌入
    embedding_single = embedder.embed_text("这是一个示例文本。")

    # 批量文本嵌入
    documents = ["文档1内容", "文档2内容", "文档3内容"]
    embedding_batch = embedder.embed_documents(documents)