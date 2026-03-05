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
        "parse_mode": "HTML"  # ← change from Markdown to HTML
    }
    response = requests.post(url, json=payload)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
    return response.ok

    # Add debug output
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")

    return response.ok

def get_orders(settings):
    local_settings = copy.deepcopy(settings)
    local_settings['verbose'] = False
    local_settings['qstats'] = False

    # Load data and compute
    data_ind = mdf.Data_Ind_Feed(local_settings).data_ind
    data, _ = data_ind
    log_history, _, bt_log_dict = compute(local_settings, data_ind)

    # Get exchange rate from last available close
    exchange_rate = data.tickers_closes['EURUSD=X'].iloc[-settings['add_days'] - 1]

    return log_history, exchange_rate


def calc_eur_amount(row, exchange_rate, settings):
    """Calculate EUR equivalent amount for an order."""
    ticker = row['ticker']
    if ticker == 'cash':
        return None  # skip cash
    mult = settings['mults'].get(ticker, 1)
    price = row['price'] if row['exectype'] == 'Stop' else 0
    eur_amount = abs(row['size']) * price * mult * exchange_rate
    return eur_amount


def format_orders_message(log_history, exchange_rate, settings, app_url):
    orders_history = log_history[log_history['event'].str.contains('Created')]
    today = datetime.date.today()

    lines = ["🔔 <b>Trading Orders Update:</b>\n"]

    def format_order_block(orders, title):
        if len(orders) > 0:
            lines.append(f"📋 <b>{title}</b>")
            for _, row in orders.iterrows():
                line = f"• <b>{row['ticker']}</b> {row['exectype']} {row['B_S']} {row['size']}"
                if row['exectype'] == "Stop":
                    line += f" @ {row['price']}"
                eur = calc_eur_amount(row, exchange_rate, settings)
                if eur is not None:
                    line += f" | {eur:,.0f}€"
                lines.append(line)
        else:
            lines.append(f"📋 <b>{title}</b>\nNo orders.")

    # Today Orders
    today_orders = orders_history[orders_history['date'] == today]
    format_order_block(today_orders, f"Today Orders {today} 00:00 (CET)")

    lines.append("")  # spacer

    # Next Orders Forecast
    orders_ahead = orders_history[orders_history['date'] > today]
    if len(orders_ahead) > 0:
        next_day = orders_ahead['date'].iloc[0]
        next_orders = orders_ahead[orders_ahead['date'] == next_day]
        format_order_block(next_orders, f"Next Orders Forecast {next_day} 00:00 (CET)")
    else:
        lines.append("🔮 <b>Next Orders Forecast</b>\nNo upcoming orders.")

    lines.append(f"\n⚠️ <i>Place orders manually with your broker.</i>")
    lines.append(f"\n🚀 <a href='{app_url}'>Open Trading App</a>")

    return "\n".join(lines)


def main():
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    app_url = "https://quanttradingapp.streamlit.app"  # ← update if different

    print("Fetching orders...")
    log_history, exchange_rate = get_orders(settings)

    message = format_orders_message(log_history, exchange_rate, settings, app_url)
    print(f"Sending alert:\n{message}")

    ok = send_telegram(token, chat_id, message)
    if ok:
        print("✅ Telegram alert sent successfully!")
    else:
        print("❌ Failed to send Telegram alert.")
