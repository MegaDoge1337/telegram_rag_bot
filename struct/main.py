from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from dotenv import load_dotenv
import codecs
import json
import os


SUMMARY_PROMPT = """Ты полезный помощник.
    User: Суммаризируй (законспектируй) текст: {content}...
    Assistant:"""


def make_struct():
  load_dotenv()
  swebhelp = codecs.open('./struct/s_webhelp.json', 'r', encoding='utf8').read()


  webhelp = codecs.open('./lc/d/rx_webhelp.json', 'r', encoding='utf8').read()
  webhelp = json.loads(webhelp)

  struct = json.loads(swebhelp)
  counter = {'processed': 0, 'total': len(webhelp)}
  for article in webhelp:
    if article['title'] in swebhelp:
      counter['processed'] += 1
      print(f'{counter["processed"]}/{counter["total"]} > {article["title"]}')
      continue

    try:
      user_prompt = PromptTemplate(
        input_variables=['title', 'content'],
        template=SUMMARY_PROMPT
      )

      llm = LlamaCpp(
          model_path=os.environ['MODEL_PATH'],
          n_ctx=4096,
          n_batch=4096,
          n_gpu_layers=30,
          n_threads=4,
          temperature=0,
          max_tokens=1024,
          seed=-1,
          verbose=False
      )

      chain = LLMChain(llm=llm, prompt=user_prompt, output_key='summary')
      result = chain({'title': article['title'], 'content': article['content']})
      struct.append({'title': article['title'], 'content': (result['summary']).replace('bot', '').rstrip().strip()})
      counter['processed'] += 1
      print(f'{counter["processed"]}/{counter["total"]} > {article["title"]}')
      print(f'---\n{(result["summary"]).replace("bot", "").rstrip().strip()}\n---')
      with open('./struct/s_webhelp.json', 'w', encoding='utf8') as file:
        json.dump(struct, file, ensure_ascii=False)
    except Exception as ex:
      print(ex)
      struct.append({'title': article['title'], 'content': None})
      counter['processed'] += 1
      print(f'{counter["processed"]}/{counter["total"]} > {article["title"]}')
      with open('./struct/s_webhelp.json', 'w', encoding='utf8') as file:
        json.dump(struct, file, ensure_ascii=False)



if __name__ == '__main__':
  make_struct()