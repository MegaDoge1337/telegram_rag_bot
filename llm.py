from langchain_community.document_transformers import LongContextReorder
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
import prompt
import os


def query_with_context(query, context):
    user_prompt = PromptTemplate(
        input_variables=['query', 'context'],
        template="""Ты полезный помощник.
        User: Ответь на вопрос `{query}` используя краткую справку - `{context}`
        Assistant:"""
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

    chain = LLMChain(llm=llm, prompt=user_prompt, output_key='answer')
    result = chain({'query': query, 'context': context})

    return result['answer']
