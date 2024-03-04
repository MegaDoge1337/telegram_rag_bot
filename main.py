from dotenv import load_dotenv
import telebot
import time
import os

import kb
import text
import llm
import vs

if not vs.is_exists():
    vs.load()
load_dotenv()
bot = telebot.TeleBot(os.environ['TELEGRAM_BOT_TOKEN'])


@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    print(call.data)


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
    chat_id = message.chat.id
    query = message.text.replace('/query', '')

    metrics = {"prompts": 0, "completions": 0}
    start_time = time.time()

    answer_message = bot.reply_to(message, text.VS_SEARCH_MESSAGE, parse_mode='HTML')
    documents = vs.similarity_search(query)

    summaries = []
    ready_summaries = 1
    total_summaries = len(documents)

    for summary, summary_metrics in llm.summarization_documents(documents):
        bot.edit_message_text(
            text.LLM_SUMMARIZATION_MESSAGE(ready_summaries, total_summaries),
            chat_id,
            answer_message.id,
            parse_mode='HTML')
        summaries.append(summary)
        metrics["prompts"] += summary_metrics["prompts"]
        metrics["completions"] += summary_metrics["completions"]
        ready_summaries += 1

    complete_summary = "\n".join(summaries)
    bot.edit_message_text(
        text.LLM_ANSWER_GENERATION_MESSAGE,
        chat_id,
        answer_message.id,
        parse_mode='HTML')

    answer_text, answer_metrics = llm.query_by_summary(query, complete_summary)
    metrics["prompts"] += answer_metrics["prompts"]
    metrics["completions"] += answer_metrics["completions"]

    end_time = time.time()
    elapsed_time = end_time - start_time

    metrics_span = text.METRICS_SPAN(metrics)
    time_span = text.TIME_SPAN(elapsed_time)
    complete_answer = text.LLM_ANSWER(metrics_span, time_span, answer_text)

    rate_answer_keyboard = kb.GET_ANSWER_RATE_KEYBOARD_MARKUP()

    bot.edit_message_text(complete_answer,
                            chat_id,
                            answer_message.id,
                            reply_markup=rate_answer_keyboard,
                            parse_mode='HTML')


while True:
    try:
        bot.polling(none_stop=True, interval=0)
    except Exception as _ex:
        print(_ex)
        time.sleep(15)
