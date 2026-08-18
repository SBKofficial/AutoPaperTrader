import os
import json
import logging
from datetime import datetime, timedelta
import pytz
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Defaults
)

warnings.filterwarnings('ignore')
load_dotenv()

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

STATE_FILE = "paper_portfolio_state.json"
LOG_FILE = "paper_trade_log.csv"

STARTING_CAPITAL = 50000.0
MAX_SLOTS = 2
MAX_RISK_PCT = 2.0          # Risk 2% of total equity per trade
TRAILING_STOP_PCT = 0.045   # 4.5% Dynamic Trailing Stop
MAX_HOLD_DAYS = 21          # 21-Day Time Stop

UNIVERSE = [
    "ABB.NS", "BSE.NS", "BEL.NS", "POLYCAB.NS", "PERSISTENT.NS",
    "DIXON.NS", "COFORGE.NS", "HAVELLS.NS", "SRF.NS", "TVSMOTOR.NS",
    "TRENT.NS", "HAL.NS", "JINDALSTEL.NS", "AUROPHARMA.NS", "LUPIN.NS",
    "CHOLAFIN.NS", "RECLTD.NS", "PFC.NS", "SIEMENS.NS", "APOLLOTYRE.NS"
]
BENCHMARK = "^NSEI"
IST = pytz.timezone('Asia/Kolkata')

# ==============================================================================
# TECHNICAL INDICATORS (PURE PANDAS - NO DEPENDENCY ISSUES)
# ==============================================================================
def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def calc_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()

def calc_roc(series: pd.Series, length: int) -> pd.Series:
    return series.pct_change(periods=length) * 100.0

# ==============================================================================
# STATE & LOGGING MANAGEMENT
# ==============================================================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {e}")
    return {
        "cash": STARTING_CAPITAL,
        "active_positions": {},
        "pending_signals": [],
        "last_run_date": None
    }

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving state file: {e}")

def log_trade(date, ticker, action, price, shares, pnl, days_held, note=""):
    record = {
        "Date": date,
        "Ticker": ticker,
        "Action": action,
        "Price": round(price, 2),
        "Shares": shares,
        "PnL": round(pnl, 2) if pnl != 0 else "-",
        "Days_Held": days_held if days_held > 0 else "-",
        "Note": note
    }
    df = pd.DataFrame([record])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

def fetch_market_data():
    start_date = (datetime.now(IST) - timedelta(days=365)).strftime('%Y-%m-%d')
    data = yf.download(UNIVERSE + [BENCHMARK], start=start_date, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data['Close']
        open_df = data['Open']
        high_df = data['High']
        low_df = data['Low']
    else:
        raise ValueError("Invalid dataframe format received from yfinance.")
    
    return close_df, open_df, high_df, low_df

# ==============================================================================
# CORE EXECUTION ENGINE
# ==============================================================================
def execute_engine_cycle():
    state = load_state()
    close_df, open_df, high_df, low_df = fetch_market_data()
    today_date = close_df.index[-1].strftime('%Y-%m-%d')
    
    logs = []

    # Calculate Current Equity
    current_equity = state["cash"]
    for ticker, pos in state["active_positions"].items():
        if ticker in close_df.columns:
            current_equity += pos["shares"] * float(close_df[ticker].iloc[-1])

    # 1. Execute Pending Signals at Today's Open (Gap-up permitted)
    if state.get("pending_signals"):
        state["pending_signals"].sort(key=lambda x: x["momentum"], reverse=True)
        remaining_signals = []
        for sig in state["pending_signals"]:
            ticker = sig["ticker"]
            if len(state["active_positions"]) >= MAX_SLOTS:
                continue

            if ticker not in open_df.columns or np.isnan(open_df[ticker].iloc[-1]):
                remaining_signals.append(sig)
                continue

            open_price = float(open_df[ticker].iloc[-1])
            initial_sl = open_price * (1 - TRAILING_STOP_PCT)
            risk_per_share = open_price - initial_sl
            
            max_account_risk = current_equity * (MAX_RISK_PCT / 100)
            max_slot_capital = current_equity / MAX_SLOTS
            
            shares_by_risk = int(max_account_risk / risk_per_share)
            if (shares_by_risk * open_price) > max_slot_capital:
                shares_to_buy = int(max_slot_capital / open_price)
            else:
                shares_to_buy = shares_by_risk

            capital_needed = shares_to_buy * open_price
            if shares_to_buy > 0 and state["cash"] >= capital_needed:
                state["cash"] -= capital_needed
                state["active_positions"][ticker] = {
                    "entry_date": today_date,
                    "entry_price": open_price,
                    "shares": shares_to_buy,
                    "highest_price": open_price,
                    "trailing_sl": initial_sl,
                    "days_held": 0
                }
                log_trade(today_date, ticker, "BUY (Open Fill)", open_price, shares_to_buy, 0.0, 0, f"Signal from {sig['signal_date']}")
                logs.append(f"🟢 *BOUGHT:* `{ticker}` | {shares_to_buy} shares @ ₹{open_price:,.2f}")
        state["pending_signals"] = remaining_signals

    # 2. Check Exits (Trailing SL & 21-Day Time Stop)
    exited_tickers = []
    for ticker, pos in list(state["active_positions"].items()):
        if ticker not in close_df.columns: continue
        
        pos["days_held"] += 1
        day_high = float(high_df[ticker].iloc[-1])
        day_low = float(low_df[ticker].iloc[-1])
        day_close = float(close_df[ticker].iloc[-1])
        
        if day_high > pos["highest_price"]:
            pos["highest_price"] = day_high
            pos["trailing_sl"] = day_high * (1 - TRAILING_STOP_PCT)
            
        if day_low <= pos["trailing_sl"]:
            exit_price = min(pos["trailing_sl"], float(open_df[ticker].iloc[-1])) if float(open_df[ticker].iloc[-1]) < pos["trailing_sl"] else pos["trailing_sl"]
            exit_val = pos["shares"] * exit_price
            pnl = exit_val - (pos["shares"] * pos["entry_price"])
            state["cash"] += exit_val
            action = "TRAILING STOP (Win)" if pnl > 0 else "STOP LOSS (-4.5%)"
            log_trade(today_date, ticker, action, exit_price, pos["shares"], pnl, pos["days_held"])
            logs.append(f"🔴 *EXITED ({action}):* `{ticker}` | PnL: ₹{pnl:,.2f} | Held: {pos['days_held']}d")
            exited_tickers.append(ticker)
        elif pos["days_held"] >= MAX_HOLD_DAYS:
            exit_price = day_close
            exit_val = pos["shares"] * exit_price
            pnl = exit_val - (pos["shares"] * pos["entry_price"])
            state["cash"] += exit_val
            action = "TIME STOP (Win)" if pnl > 0 else "TIME STOP (Loss)"
            log_trade(today_date, ticker, action, exit_price, pos["shares"], pnl, pos["days_held"])
            logs.append(f"⏰ *EXITED ({action}):* `{ticker}` | PnL: ₹{pnl:,.2f} | Held: {pos['days_held']}d")
            exited_tickers.append(ticker)

    for ticker in exited_tickers:
        del state["active_positions"][ticker]

    # 3. Evening Screener (Queue Buy Signals for Tomorrow's Open)
    nifty_close = close_df[BENCHMARK].dropna()
    nifty_200_sma = float(calc_sma(nifty_close, 200).iloc[-1])
    nifty_roc_63 = float(calc_roc(nifty_close, 63).iloc[-1])
    macro_bullish = float(nifty_close.iloc[-1]) > nifty_200_sma
    state["pending_signals"] = []

    if macro_bullish:
        for ticker in UNIVERSE:
            if ticker in state["active_positions"]: continue
            if ticker not in close_df.columns: continue
            df = pd.DataFrame({'Close': close_df[ticker]}).dropna()
            if len(df) < 200: continue
            
            df['9_EMA'] = calc_ema(df['Close'], 9)
            df['21_EMA'] = calc_ema(df['Close'], 21)
            df['200_EMA'] = calc_ema(df['Close'], 200)
            df['ROC_63'] = calc_roc(df['Close'], 63)
            
            df['9_above_21'] = df['9_EMA'] > df['21_EMA']
            prev_day = df['9_above_21'].shift(1).fillna(False).astype(bool)
            df['Crossover'] = df['9_above_21'] & (~prev_day)
            
            latest = df.iloc[-1]
            if latest['Crossover'] and (latest['Close'] > latest['200_EMA']) and (latest['ROC_63'] > nifty_roc_63):
                state["pending_signals"].append({
                    "ticker": ticker,
                    "signal_date": today_date,
                    "momentum": float(latest['ROC_63']),
                    "close_price": float(latest['Close'])
                })
                logs.append(f"⚡ *NEW SIGNAL:* `{ticker}` (ROC: {latest['ROC_63']:.1f}% vs Nifty: {nifty_roc_63:.1f}%)")

    state["last_run_date"] = today_date
    save_state(state)
    return logs

# ==============================================================================
# TELEGRAM COMMAND HANDLERS
# ==============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 *Auto Paper Trading Bot Active*\n\n"
        "📈 *Strategy Rules:*\n"
        "• Strategy: `9/21 EMA Momentum Cross`\n"
        "• Universe: `Top 20 Liquid Midcaps`\n"
        "• Macro Filter: `Nifty > 200 SMA`\n"
        "• Execution: `Next-day Open (Gap-up permitted)`\n"
        "• Risk per Trade: `2.0% of Total Capital`\n"
        "• Max Active Slots: `2 concurrent positions`\n"
        "• Exit Rules: `4.5% Trailing SL | 21-Day Time Stop`\n\n"
        "📌 *Available Commands:*\n"
        "/portfolio - Market status, active holdings & live P&L\n"
        "/balance - Available cash, invested capital & net ROI\n"
        "/tradelog - Last 5 completed trade executions\n"
        "/scan - Trigger manual market scan & order execution"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    close_df, _, _, _ = fetch_market_data()
    
    invested_val = 0.0
    for ticker, pos in state["active_positions"].items():
        if ticker in close_df.columns:
            invested_val += pos["shares"] * float(close_df[ticker].iloc[-1])
            
    total_equity = state["cash"] + invested_val
    pnl = total_equity - STARTING_CAPITAL
    roi = (pnl / STARTING_CAPITAL) * 100

    msg = (
        "💳 *ACCOUNT BALANCE & CAPITAL*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 *Available Cash:* `₹{state['cash']:,.2f}`\n"
        f"📊 *Invested Capital:* `₹{invested_val:,.2f}`\n"
        f"📈 *Total Equity:* `₹{total_equity:,.2f}`\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *Net Profit/Loss:* `₹{pnl:+,.2f}` (`{roi:+.2f}%`)\n"
        f"💼 *Slots Occupied:* `{len(state['active_positions'])} / {MAX_SLOTS}`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    close_df, _, _, _ = fetch_market_data()

    nifty_close = close_df[BENCHMARK].dropna()
    nifty_last = float(nifty_close.iloc[-1])
    nifty_200_sma = float(calc_sma(nifty_close, 200).iloc[-1])
    is_bullish = nifty_last > nifty_200_sma
    macro_icon = "🟢 BULLISH (> 200 SMA)" if is_bullish else "🔴 BEARISH (< 200 SMA)"

    msg = [
        "📊 *PORTFOLIO & MARKET SITUATION*",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"🏛 *Nifty 50:* `₹{nifty_last:,.2f}`",
        f"📏 *200 SMA:* `₹{nifty_200_sma:,.2f}`",
        f"🚦 *Macro Filter:* *{macro_icon}*",
        "━━━━━━━━━━━━━━━━━━━━━",
        "📦 *CURRENT HOLDINGS:*"
    ]

    if not state["active_positions"]:
        msg.append("_(No active positions. 100% capital in cash.)_")
    else:
        for ticker, pos in state["active_positions"].items():
            curr_px = float(close_df[ticker].iloc[-1]) if ticker in close_df.columns else pos['entry_price']
            unrealized = (curr_px - pos['entry_price']) * pos['shares']
            pnl_pct = ((curr_px - pos['entry_price']) / pos['entry_price']) * 100
            icon = "🟢" if unrealized >= 0 else "🔴"

            msg.append(
                f"\n{icon} *{ticker}* ({pos['shares']} shares)\n"
                f" • Entry: `₹{pos['entry_price']:,.2f}` | Last: `₹{curr_px:,.2f}`\n"
                f" • Trailing SL: `₹{pos['trailing_sl']:,.2f}`\n"
                f" • Unrealized P&L: `₹{unrealized:+,.2f}` (`{pnl_pct:+.2f}%`)\n"
                f" • Duration: `{pos['days_held']}/{MAX_HOLD_DAYS} days`"
            )

    if state.get("pending_signals"):
        msg.append("\n━━━━━━━━━━━━━━━━━━━━━")
        msg.append("⚡ *QUEUED FOR NEXT OPEN:*")
        for s in state["pending_signals"]:
            msg.append(f" • `{s['ticker']}` | Prev Close: `₹{s['close_price']:,.2f}` | RS: `{s['momentum']:.1f}%`")

    await update.message.reply_text("\n".join(msg), parse_mode="Markdown")

async def tradelog_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("📂 *No trades recorded yet in log file.*", parse_mode="Markdown")
        return

    try:
        df = pd.read_csv(LOG_FILE)
        exits = df[df['Action'] != 'BUY (Open Fill)'].tail(5)

        if exits.empty:
            await update.message.reply_text("📂 *No closed trade exits recorded yet.*", parse_mode="Markdown")
            return

        msg = [
            "📜 *PAST 5 COMPLETED TRADES*",
            "━━━━━━━━━━━━━━━━━━━━━"
        ]

        for _, row in exits.iterrows():
            pnl_val = float(str(row['PnL']).replace("₹", "").replace(",", "")) if row['PnL'] != "-" else 0.0
            icon = "🟢" if pnl_val > 0 else "🔴"
            
            msg.append(
                f"{icon} *{row['Ticker']}* | `{row['Date']}`\n"
                f" • Action: `{row['Action']}`\n"
                f" • Exit Price: `₹{float(row['Price']):,.2f}` | Qty: `{row['Shares']}`\n"
                f" • Days Held: `{row['Days_Held']} days`\n"
                f" • Realized P&L: *₹{pnl_val:+,.2f}*\n"
            )

        await update.message.reply_text("\n".join(msg), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error reading trade log: {e}")
        await update.message.reply_text("⚠️ *Error reading trade log file.*", parse_mode="Markdown")

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ *Running strategy screener & processing triggers...*", parse_mode="Markdown")
    try:
        logs = execute_engine_cycle()
        summary = "✅ *Scan Complete - Actions Taken:*\n\n" + "\n".join(logs) if logs else "✅ *Scan Complete:* No new signals or exit triggers today."
        await update.message.reply_text(summary, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error executing scan cycle: {e}")
        await update.message.reply_text(f"⚠️ *Scan Error:* `{str(e)}`", parse_mode="Markdown")

# ==============================================================================
# AUTOMATED SCHEDULED JOBS
# ==============================================================================
async def scheduled_daily_run(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Executing scheduled daily market cycle...")
    logs = execute_engine_cycle()
    if ADMIN_CHAT_ID and logs:
        try:
            alert = "🔔 *Automated Paper Trading Execution Update:*\n\n" + "\n".join(logs)
            await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=alert, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed to send admin notification: {e}")

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN is not set. Please add it to your environment variables.")

    defaults = Defaults(tzinfo=IST)
    app = ApplicationBuilder().token(BOT_TOKEN).defaults(defaults).build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("balance", balance_command))
    app.add_handler(CommandHandler("tradelog", tradelog_command))
    app.add_handler(CommandHandler("scan", scan_command))

    # Automated Scheduling (Mon-Fri)
    job_queue = app.job_queue
    if job_queue:
        from datetime import time
        job_queue.run_daily(
            scheduled_daily_run,
            time=time(hour=9, minute=18, tzinfo=IST),
            days=(0, 1, 2, 3, 4)
        )
        job_queue.run_daily(
            scheduled_daily_run,
            time=time(hour=15, minute=35, tzinfo=IST),
            days=(0, 1, 2, 3, 4)
        )
        logger.info("Scheduled jobs registered for 09:18 AM and 03:35 PM IST.")

    logger.info("Bot is starting polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
