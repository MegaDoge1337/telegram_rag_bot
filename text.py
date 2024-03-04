INFO_MESSAGE = """• 🦜 Решение работает на базе фреймворка <a href=\"https://www.langchain.com/\">LangChain</a> версии 0.1.5 
• 🚀 LLM: <a href=\"https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf\">IlyaGusev/saiga_mistral_7b_gguf</a>
• 🔍 Векторное хранилище: <a href=\"https://docs.docarray.org/\">DocArray</a> (алгоритм поиска <a href=\"https://www.pinecone.io/learn/series/faiss/hnsw/\">HNSW</a>)
• 📦 Содержит > 3000 статей из справки по СЭД <a href=\"https://www.directum.ru/products/directum\">Directum RX</a> версии 4.7
• ❓ Бот работает в режиме QA (вопрос-ответ)"""

VS_SEARCH_MESSAGE = """🗃️ Поиск релевантных документов..."""


def LLM_SUMMARIZATION_MESSAGE(ready, total):
    return f"""📑 Суммаризация документов [{ready}/{total}]..."""


LLM_ANSWER_GENERATION_MESSAGE = """💭 Генерация ответа..."""


def METRICS_SPAN(metrics):
    return f"""💬 <code>Токенов отправлено/сгенерировано: {metrics["prompts"]}/{metrics["completions"]}</code>"""


def TIME_SPAN(elapsed_time):
    return f"""⌛ <code>Времени прошло: {int(elapsed_time) // 60} мин. {int(elapsed_time) % 60} сек.</code>"""


def LLM_ANSWER(metrics_span, time_span, answer_text):
    return f"""ℹ️ {answer_text}\n\n{metrics_span}\n{time_span}"""
