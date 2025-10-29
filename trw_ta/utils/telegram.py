import requests

def send_telegram_message(bot_token: str, chat_id: str, message: str):
    """
    Send a message to a Telegram chat using the Telegram Bot API.

    Parameters
    ----------
    bot_token : str
        The API token of your Telegram bot. This token is obtained from @BotFather.
    chat_id : str
        The unique identifier of the target chat or the username of the target channel (prefixed with '@').
    message : str
        The text message to be sent.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message})
    except Exception as e:
        print(f"Telegram error: {e}")
