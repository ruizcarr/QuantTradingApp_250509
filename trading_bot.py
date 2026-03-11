import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import datetime
import copy
import pytz

from config.trading_settings import settings
import Market_Data_Feed as mdf
from Trading_Markowitz import compute, process_log_data

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_EURIBOR = 0.001


def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, json=payload)
    print(f"Status code: {resp.status_code}")
    print(f"Response: {resp.text}")
    return resp.ok


def get_orders(settings):
    local_settings = copy.deepcopy(settings)
    local_settings['verbose'] = False
    local_settings['qstats'] = False

    data_ind = mdf.Data_Ind_Feed(local_settings).data_ind
    data, _ = data_ind
    log_history, _, bt_log_dict = compute(local_settings, data_ind)

    # Process log to get eod positions
    eod_log_history, trading_history = process_log_data(log_history, local_settings)

    # Exchange rate
    exchange_rate = data.tickers_closes['EURUSD=X'].iloc[-local_settings['add_days'] - 1]

    # Today
    today = datetime.date.today()
    eod_today = eod_log_history[eod_log_history.index.date <= today]

    # Euribor rate
    from Market_Data_Feed import get_euribor_1y_daily
    euribor_df = get_euribor_1y_daily()
    euribor_rate = euribor_df['Euribor'].iloc[-1]

    # Cash info
    cash_eur_series = bt_log_dict['cash_eur']
    eod_today_cash = cash_eur_series[cash_eur_series.index.date <= today]

    current_cash_eur = float(eod_today_cash.iloc[-1]) if euribor_rate > MIN_EURIBOR else 0.0
    previous_cash_eur = float(eod_today_cash.iloc[-2]) if euribor_rate > MIN_EURIBOR else 0.0

    cash_eur_change = None
    if abs(current_cash_eur - previous_cash_eur) > 1:
        cash_eur_change = current_cash_eur - previous_cash_eur

    cash_info = {
        'eur': current_cash_eur,
        'euribor': euribor_rate,
        'eur_change': cash_eur_change,
    }

    # Current positions
    last_row = eod_today.iloc[-1]
    portfolio_value_eur = last_row['portfolio_value_eur']

    positions_info = []
    for ticker in eod_log_history.columns:
        if ticker not in settings['tickers'] and ticker != 'cash':
            continue
        if ticker == 'cash':
            continue
        contracts = int(last_row[ticker])
        if contracts == 0:
            continue
        last_price = data.tickers_closes[ticker].iloc[-local_settings['add_days'] - 1]
        mult = local_settings['mults'].get(ticker, 1)
        eur_value = contracts * last_price * mult / exchange_rate
        pct = eur_value / portfolio_value_eur * 100
        positions_info.append({
            'ticker': ticker,
            'contracts': contracts,
            'eur_value': eur_value,
            'pct': pct,
        })

    # Portfolio performance
    total_eur = sum(p['eur_value'] for p in positions_info)
    total_pct = total_eur / portfolio_value_eur * 100

    prev_portfolio_eur = eod_today.iloc[-2]['portfolio_value_eur']
    daily_change_eur = portfolio_value_eur - prev_portfolio_eur
    daily_change_pct = daily_change_eur / prev_portfolio_eur * 100

    weekly_eur    = eod_today.iloc[-6]['portfolio_value_eur']   if len(eod_today) >= 6   else None
    monthly_eur   = eod_today.iloc[-23]['portfolio_value_eur']  if len(eod_today) >= 23  else None
    yearly_eur    = eod_today.iloc[-253]['portfolio_value_eur'] if len(eod_today) >= 253 else None

    weekly_change_eur  = portfolio_value_eur - weekly_eur  if weekly_eur  is not None else None
    weekly_change_pct  = weekly_change_eur  / weekly_eur   * 100 if weekly_eur  is not None else None
    monthly_change_eur = portfolio_value_eur - monthly_eur if monthly_eur is not None else None
    monthly_change_pct = monthly_change_eur / monthly_eur  * 100 if monthly_eur is not None else None
    yearly_change_eur  = portfolio_value_eur - yearly_eur  if yearly_eur  is not None else None
    yearly_change_pct  = yearly_change_eur  / yearly_eur   * 100 if yearly_eur  is not None else None

    portfolio_info = {
        'total_eur':          total_eur,
        'total_pct':          total_pct,
        'portfolio_value_eur': portfolio_value_eur,
        'daily_change_pct':   daily_change_pct,
        'daily_change_eur':   daily_change_eur,
        'weekly_change_pct':  weekly_change_pct,
        'weekly_change_eur':  weekly_change_eur,
        'monthly_change_pct': monthly_change_pct,
        'monthly_change_eur': monthly_change_eur,
        'yearly_change_pct':  yearly_change_pct,
        'yearly_change_eur':  yearly_change_eur,
    }

    return log_history, exchange_rate, cash_info, positions_info, portfolio_info


def calc_eur_amount(row, exchange_rate, settings):
    ticker = row['ticker']
    mult = settings['mults'].get(ticker, 1)
    price = row['price'] if row['exectype'] == 'Stop' else 0
    eur_amount = abs(row['size']) * price * mult / exchange_rate
    return eur_amount


def format_orders_message(log_history, exchange_rate, cash_info, positions_info, portfolio_info, settings, app_url):
    orders_history = log_history[log_history['event'].str.contains('Created')]
    orders_history = orders_history[orders_history['ticker'] != 'cash']

    today = datetime.date.today()
    tz = pytz.timezone('Europe/Madrid')
    now = datetime.datetime.now(tz).strftime("%H:%M")

    lines = []
    lines.append(f"🔔 <b>Trading Orders Update</b>")
    lines.append(f"{today} {now}")
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
                    line += f" | {eur:,.0f}€"
                lines.append(line)
        else:
            lines.append(f"{title}\nNo orders.")

    # Today Orders
    today_orders = orders_history[orders_history['date'] == today]
    format_order_block(today_orders, f"<b>Today Orders</b>\n🕒 {today} 00:00")

    lines.append("")

    # Upcoming Orders
    orders_ahead = orders_history[orders_history['date'] > today]
    if len(orders_ahead) > 0:
        next_day = orders_ahead['date'].iloc[0]
        next_orders = orders_ahead[orders_ahead['date'] == next_day]
        format_order_block(next_orders, f"<b>Upcoming Orders</b>\n🕒 {next_day} 00:00")
    else:
        lines.append("<b>Upcoming Orders</b>\nNo upcoming orders.")

    # Cash
    lines.append("")
    cash_line = f"💰 <b>Cash:</b> {cash_info['eur']:,.0f}€ @ Euribor {cash_info['euribor']:.2%}"
    lines.append(cash_line)
    #if cash_info['eur_change'] is not None:
    #    change_icon = "📈" if cash_info['eur_change'] > 0 else "📉"
    #    lines.append(f"   {change_icon} Change: {cash_info['eur_change']:+,.0f}€")

    # Positions
    if positions_info:
        lines.append("")
        lines.append(f"📊 <b>Current Positions:</b>")
        for pos in positions_info:
            line = f"   <b>{pos['ticker']}</b> | {pos['contracts']} | {pos['pct']:.0f}% | {pos['eur_value']:,.0f}€"
            lines.append(line)
        lines.append(f"   <b>Exposition</b> | {portfolio_info['total_pct']:.0f}% | {portfolio_info['total_eur']:,.0f}€")
        lines.append(f"   💼 <b>Portfolio: {portfolio_info['portfolio_value_eur']:,.0f}€</b>")
        lines.append("")

        d_icon = "🟢" if portfolio_info['daily_change_pct'] > 0 else "🔴"
        lines.append(f"   {d_icon} Daily:   {portfolio_info['daily_change_pct']:+.2f}% | {portfolio_info['daily_change_eur']:+,.0f}€")

        if portfolio_info['weekly_change_pct'] is not None:
            w_icon = "🟢" if portfolio_info['weekly_change_pct'] > 0 else "🔴"
            lines.append(f"   {w_icon} Weekly:  {portfolio_info['weekly_change_pct']:+.2f}% | {portfolio_info['weekly_change_eur']:+,.0f}€")

        if portfolio_info['monthly_change_pct'] is not None:
            m_icon = "🟢" if portfolio_info['monthly_change_pct'] > 0 else "🔴"
            lines.append(f"   {m_icon} Monthly: {portfolio_info['monthly_change_pct']:+.2f}% | {portfolio_info['monthly_change_eur']:+,.0f}€")

        if portfolio_info['yearly_change_pct'] is not None:
            y_icon = "🟢" if portfolio_info['yearly_change_pct'] > 0 else "🔴"
            lines.append(f"   {y_icon} Yearly:  {portfolio_info['yearly_change_pct']:+.2f}% | {portfolio_info['yearly_change_eur']:+,.0f}€")

    lines.append("")
    #lines.append("⚠️ Place orders manually with your broker.")
    lines.append(f"🚀 Open Trading App: {app_url}")

    return "\n".join(lines)


def main():
    token   = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    app_url = "https://quanttradingapp.streamlit.app"

    try:
        print("Step 1: Fetching orders...")
        log_history, exchange_rate, cash_info, positions_info, portfolio_info = get_orders(settings)
        print("Step 2: Orders fetched OK")
        print(f"Cash info: {cash_info}")

        message = format_orders_message(
            log_history, exchange_rate, cash_info, positions_info, portfolio_info, settings, app_url
        )
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