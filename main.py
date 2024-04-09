from dotenv import load_dotenv
import telebot
import time
import os

import text
import llm
import vs

if not vs.is_exists():
    vs.load()
load_dotenv(override=True)
bot = telebot.TeleBot(os.environ['TELEGRAM_BOT_TOKEN'])


@bot.message_handler(commands=['start'])
def start_bot(message):
    chat_id = message.chat.id
    start_message_id = message.id
    bot.delete_message(chat_id, start_message_id)


@bot.message_handler(commands=['info'])
def get_info(message):
    chat_id = message.chat.id
    command_message_id = message.id
    bot.delete_message(chat_id, command_message_id)
    bot.send_message(chat_id, text.INFO_MESSAGE, parse_mode='HTML')


@bot.message_handler(commands=['query'])
def make_llm_query(message):
    start_time = time.time()

    chat_id = message.chat.id
    query = message.text.replace('/query', '')
    answer_message = bot.reply_to(message, text.VS_SEARCH_MESSAGE, parse_mode='HTML')
    documents = vs.similarity_search(query)

    summaries = []
    ready_summaries = 1
    total_summaries = len(documents)

    for summary in llm.summarization_documents(documents):
        bot.edit_message_text(
            text.LLM_SUMMARIZATION_MESSAGE(ready_summaries, total_summaries),
            chat_id,
            answer_message.id,
            parse_mode='HTML')
        summaries.append(summary)
        ready_summaries += 1

    complete_summary = "\n".join(summaries)
    complete_summary = complete_summary.replace('bot', '').rstrip().strip()
    bot.edit_message_text(
        text.LLM_ANSWER_GENERATION_MESSAGE,
        chat_id,
        answer_message.id,
        parse_mode='HTML')

    answer_text = llm.query_by_summary(query, complete_summary)
    answer_text = answer_text.replace('bot', '').rstrip().strip()

    end_time = time.time()

    elapsed_time = end_time - start_time
    time_span = text.TIME_SPAN(elapsed_time)

    complete_answer = text.LLM_ANSWER(time_span, answer_text)


    bot.edit_message_text(complete_answer,
                            chat_id,
                            answer_message.id,
                            parse_mode='HTML')


while True:
    try:
        bot.polling(none_stop=True, interval=0)
    except Exception as _ex:
        print(_ex)
        time.sleep(15)
#bot.polling(none_stop=True, interval=0)