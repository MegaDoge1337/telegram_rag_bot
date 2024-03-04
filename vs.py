from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import DocArrayHnswSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import codecs
import json
import os


def is_exists():
    return os.path.exists('./lc/v')


def load():
    print('[vector store loader] started...')
    folder = './lc/d'
    documents = os.listdir(folder)
    for document in documents:
        if '.json' not in document:
            print(f'[vector store loader] document `{folder}/{document}` not is `.json` format - ignored')
            continue
        print(f'[vector store loader] document `{folder}/{document}` is `.json` format - processing')
        data = codecs.open(f'{folder}/{document}', 'r', 'utf_8')
        data = json.load(data)
        data = pd.DataFrame(data)
        loader = DataFrameLoader(data, page_content_column='content')
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder()
        split_documents = text_splitter.split_documents(loader.load())
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        counter = {'processed': 0, 'all': len(split_documents)}
        for split_document in split_documents:
            counter['processed'] += 1
            print(f"[vector store loader] <{counter['processed']}/{counter['all']}> - {split_document}")
            DocArrayHnswSearch.from_documents([split_document], embeddings, work_dir="./lc/v/", n_dim=768)
    print('[vector store loader] finished...')


def similarity_search(query):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    vs = DocArrayHnswSearch.from_params(embedding=embeddings, work_dir="./lc/v/", n_dim=768)
    return vs.similarity_search(query)
