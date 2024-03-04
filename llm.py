from langchain_community.document_transformers import LongContextReorder
from openai import OpenAI
import prompt
import openai
import os


def _configurate_openai():
    openai.api_type = os.environ['OPENAI_API_TYPE']
    openai.base_url = os.environ['OPENAI_API_BASE']
    openai.api_key = os.environ['OPENAI_API_KEY']


def summarization_documents(documents):
    _configurate_openai()
    reordering = LongContextReorder()
    reordered_documents = reordering.transform_documents(documents)
    metrics = {"prompts": 0, "completions": 0}
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    for document in reordered_documents:
        document_meta_title = document.metadata['title'].split(' > ')
        title = document_meta_title[len(document_meta_title) - 1]
        path = "\\".join(document_meta_title)

        content = document.page_content

        completion = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": prompt.SYSTEM_PROMPT()},
                {"role": "user", "content": prompt.SUMMARY_PROMPT(content, title=title, path=path)}
            ],
            temperature=0.3,
            max_tokens=2000,
            seed=42,
            top_p=0.1
        )
        summary = completion.choices[0].message.content
        metrics['prompts'] += completion.usage.prompt_tokens
        metrics['completions'] += completion.usage.completion_tokens
        yield summary, metrics


def query_by_summary(query, summary):
    _configurate_openai()
    metrics = {"prompts": 0, "completions": 0}
    client = OpenAI(base_url=openai.base_url, api_key=openai.api_key)
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": prompt.SYSTEM_PROMPT()},
            {"role": "user", "content": prompt.QUERY_BY_SUMMARY_PROMPT(query, summary)}
        ],
        temperature=0.2,
        max_tokens=1000,
        seed=42,
        top_p=0
    )

    metrics['prompts'] += completion.usage.prompt_tokens
    metrics['completions'] += completion.usage.completion_tokens
    return completion.choices[0].message.content, metrics

