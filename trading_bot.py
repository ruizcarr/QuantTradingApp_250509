import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import datetime
import copy

from config.trading_settings import settings
import Market_Data_Feed as mdf
from Trading_Markowitz import compute, process_log_data


def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    response = requests.post(url, json=payload)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
    return response.ok


def get_orders(settings):
    local_settings = copy.deepcopy(settings)
    local_settings['verbose'] = False
    local_settings['qstats'] = False

    data_ind = mdf.Data_Ind_Feed(local_settings).data_ind
    data, _ = data_ind
    log_history, _, bt_log_dict = compute(local_settings, data_ind)

    exchange_rate = data.tickers_closes['EURUSD=X'].iloc[-local_settings['add_days'] - 1]

    return log_history, exchange_rate


def calc_eur_amount(row, exchange_rate, settings):
    ticker = row['ticker']
    mult = settings['mults'].get(ticker, 1)
    price = row['price'] if row['exectype'] == 'Stop' else 0
    eur_amount = abs(row['size']) * price * mult * exchange_rate
    return eur_amount


def format_orders_message(log_history, exchange_rate, settings, app_url):
    orders_history = log_history[log_history['event'].str.contains('Created')]
    today = datetime.date.today()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")

    lines = []
    lines.append(f"🔔 <b>Trading Orders Update</b>")
    lines.append(f"🕒 {today} {now} UTC")
    lines.append("")

    def format_order_block(orders, title):
        if len(orders) > 0:
            lines.append(title)
            for _, row in orders.iterrows():
                bs_icon = "🟢" if row['B_S'] == 'Buy' else "🔴"
                line = f"{bs_icon} <b>{row['ticker']}</b> {row['exectype']} {row['B_S']} {row['size']}"
                if row['exectype'] == "Stop":
                    line += f" @ {row['price']}"
                eur = calc_eur_amount(row, exchange_rate, settings)
                if eur is not None and eur > 0:
                    line += f" | {eur:,.0f} EUR"
                lines.append(line)
        else:
            lines.append(f"{title}\nNo orders.")

    # Today Orders
    today_orders = orders_history[orders_history['date'] == today]
    format_order_block(today_orders, f"📋 <b>Today Orders</b> {today} 00:00")

    lines.append("")

    # Next Orders Forecast
    orders_ahead = orders_history[orders_history['date'] > today]
    if len(orders_ahead) > 0:
        next_day = orders_ahead['date'].iloc[0]
        next_orders = orders_ahead[orders_ahead['date'] == next_day]
        format_order_block(next_orders, f"🔮 <b>Next Orders Forecast</b> {next_day} 00:00")
    else:
        lines.append("🔮 <b>Next Orders Forecast</b>\nNo upcoming orders.")

    lines.append("")
    lines.append("⚠️ Place orders manually with your broker.")
    lines.append(f"🚀 Open Trading App: {app_url}")

    return "\n".join(lines)


def main():
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    app_url = "https://quanttradingapp.streamlit.app"  # <- update if different

    try:
        print("Step 1: Fetching orders...")
        log_history, exchange_rate = get_orders(settings)
        print("Step 2: Orders fetched OK")

        message = format_orders_message(log_history, exchange_rate, settings, app_url)
        print(f"Step 3: Message formatted OK")
        print(f"Message:\n{message}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        send_telegram(token, chat_id, f"⚠️ Trading bot error: {str(e)}")
        return

    ok = send_telegram(token, chat_id, message)
    if ok:
        print("Telegram alert sent successfully!")
    else:
        print("Failed to send Telegram alert.")


if __name__ == "__main__":
    main()
