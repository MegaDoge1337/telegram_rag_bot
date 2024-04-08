from langchain_community.document_transformers import LongContextReorder
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
import prompt
import os


def summarization_documents(documents):
    reordering = LongContextReorder()
    reordered_documents = reordering.transform_documents(documents)
    for document in reordered_documents:
        document_meta_title = document.metadata['title'].split(' > ')
        title = document_meta_title[len(document_meta_title) - 1]
        path = "\\".join(document_meta_title)
        content = document.page_content
        user_prompt = PromptTemplate(
            input_variables=['title', 'path', 'content'],
            template=prompt.SUMMARY_PROMPT
        )

        llm = LlamaCpp(
            model_path=os.environ['MODEL_PATH'],
            n_ctx=4096,
            n_batch=4096,
            n_gpu_layers=-1,
            n_threads=4,
            temperature=0,
            max_tokens=2000,
            top_p=0.1,
            seed=42
        )

        chain = LLMChain(llm=llm, prompt=user_prompt, output_key='summary')
        result = chain({'title': title, 'path': path, 'content': content})
        yield result['summary']


def query_by_summary(query, summary):
    user_prompt = PromptTemplate(
        input_variables=['summary', 'query'],
        template=prompt.QUERY_BY_SUMMARY_PROMPT
    )

    llm = LlamaCpp(
        model_path=os.environ['MODEL_PATH'],
        n_ctx=4096,
        n_batch=4096,
        n_gpu_layers=-1,
        n_threads=4,
        temperature=0.5,
        max_tokens=500,
        top_p=0.95,
        seed=42
    )

    chain = LLMChain(llm=llm, prompt=user_prompt, output_key='answer')
    result = chain({'summary': summary, 'query': query})

    return result['answer']

