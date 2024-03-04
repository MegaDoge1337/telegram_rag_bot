from telebot import types


def GET_ANSWER_RATE_KEYBOARD_MARKUP():
    markup = types.InlineKeyboardMarkup()

    like_btn = types.InlineKeyboardButton(text='👍', callback_data='like')
    dislike_btn = types.InlineKeyboardButton(text='👎', callback_data='dislike')
    markup.row_width = 2
    markup.add(like_btn, dislike_btn)

    return markup
