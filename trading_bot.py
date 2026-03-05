import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from config.trading_settings import settings
import Market_Data_Feed as mdf
from Trading_Markowitz import compute, process_log_data
import datetime
import pandas as pd
import copy

def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    return response.ok

def get_orders(settings):
    local_settings = copy.deepcopy(settings)
    local_settings['verbose'] = False
    local_settings['qstats'] = False

    # Load data and compute
    data_ind = mdf.Data_Ind_Feed(local_settings).data_ind
    data, _ = data_ind
    log_history, _, bt_log_dict = compute(local_settings, data_ind)

    # Process log
    eod_log_history, trading_history = process_log_data(log_history, local_settings)

    # Get today's orders
    today = datetime.date.today()
    orders_history = log_history[log_history['event'].str.contains('Created')]
    today_orders = orders_history[orders_history['date'] == today]

    return today_orders

def format_orders_message(orders):
    if len(orders) == 0:
        return None  # No message needed

    lines = ["🔔 *Trading Orders for Today:*\n"]
    for _, row in orders.iterrows():
        line = f"• *{row['ticker']}* {row['exectype']} {row['B_S']} {row['size']}"
        if row['exectype'] == "Stop":
            line += f" @ {row['price']}"
        lines.append(line)

    lines.append(f"\n📅 {datetime.date.today()} 22:00 CET")
    lines.append("⚠️ _Please place orders manually with your broker._")

    return "\n".join(lines)

def main():
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    print("Fetching orders...")
    orders = get_orders(settings)

    message = format_orders_message(orders)

    if message is None:
        print("No orders today — sending no alert.")
        # Optionally notify anyway:
        send_telegram(token, chat_id, "✅ *No orders today.* Keep current positions.")
    else:
        print(f"Sending alert:\n{message}")
        ok = send_telegram(token, chat_id, message)
        if ok:
            print("✅ Telegram alert sent successfully!")
        else:
            print("❌ Failed to send Telegram alert.")

if __name__ == "__main__":
    main()