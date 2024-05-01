from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from dotenv import load_dotenv
from typing import List
import codecs
import json
import time
import os
import llm
import vs



def evaluate():
  load_dotenv()
  dataset_text = codecs.open('./evals/dataset.json', 'r', encoding='utf8').read()
  dataset_json = json.loads(dataset_text)

  counter = 1
  for row in dataset_json:
    print(counter, len(dataset_json))
    query = row['question'][0]

    vs_time_start = time.time()
    documents = vs.similarity_search(query)
    context = "\n".join(list(map(lambda x: x.page_content, documents)))
    vs_time_end = time.time()

    row['contexts'] = [[context]]

    llm_time_start = time.time()
    answer = llm.query_with_context(query, context)
    llm_time_end = time.time()

    row['answer'] = [answer.rstrip().strip()]
    row['stime'] = f'{int(vs_time_end - vs_time_start) // 60} мин. {int(vs_time_end - vs_time_start) % 60} сек.'
    row['atime'] = f'{int(llm_time_end - llm_time_start) // 60} мин. {int(llm_time_end - llm_time_start) % 60} сек.'
    counter += 1
  with open('./evals/result.json', 'w', encoding='utf8') as file:
    json.dump(dataset_json, file, ensure_ascii=False)


if __name__ == '__main__':
  if not vs.is_exists():
    vs.load()
  evaluate()