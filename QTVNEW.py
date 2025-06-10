# -*- coding: utf-8 -*-
import requests
import json
import csv
import os
from datetime import datetime
import asyncio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
import time
import logging
from fastapi import FastAPI
import uvicorn
from threading import Thread

# --- Setup Logging ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI for Render Health Check ---
app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "Bot is running"}

# --- Load Configuration ---
CONFIG_FILE = "config.json"
AUTHORIZED_CHATS_FILE = "authorized_chats.json"

def create_default_config():
    default_config = {
        "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", "7832972819:AAH0pgvibc9vxgyjeyRtwuRQYFrQV8eBJrI"),
        "ALLOWED_CHAT_ID": os.getenv("ALLOWED_CHAT_ID", "-1002554202438"),
        "VALID_KEY": os.getenv("VALID_KEY", "10092006"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY", "af9b016f3f044a6f84453bbe1a526f0b"),
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        logger.warning("Created default config.json with environment or hardcoded values.")
    except Exception as e:
        logger.error(f"Error creating default config.json: {e}")
    return default_config

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    TELEGRAM_TOKEN = config.get('TELEGRAM_TOKEN')
    ALLOWED_CHAT_ID = config.get('ALLOWED_CHAT_ID')
    VALID_KEY = config.get('VALID_KEY', '10092006')
    NEWS_API_KEY = config.get('NEWS_API_KEY', 'YOUR_NEWS_API_KEY')
    if not TELEGRAM_TOKEN or not ALLOWED_CHAT_ID:
        raise ValueError("TELEGRAM_TOKEN or ALLOWED_CHAT_ID missing in config")
except json.JSONDecodeError as e:
    logger.error(f"config.json is malformed: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

# --- Constants ---
KRAKEN_OHLC_URL = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1'
HISTORY_FILE = "price_history.csv"
MAX_HISTORY = 100
MIN_ANALYSIS_DURATION = 30
MIN_REST_DURATION = 0

# --- Global Variables ---
price_history = []
volume_history = []
high_history = []
low_history = []
open_history = []
support_level = None
resistance_level = None
is_analyzing = False
set_up_jobs = {}
authorized_chats = {}

# --- Save and Load Authorized Chats ---
def save_authorized_chats():
    try:
        with open(AUTHORIZED_CHATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(authorized_chats, f)
    except Exception as e:
        logger.error(f"Error saving authorized_chats: {e}")

def load_authorized_chats():
    global authorized_chats
    try:
        if os.path.exists(AUTHORIZED_CHATS_FILE):
            with open(AUTHORIZED_CHATS_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                authorized_chats = {str(k): v for k, v in loaded.items()}
                if len(authorized_chats) > 1:
                    recent_chat = max(
                        {k: v for k in authorized_chats if k != ALLOWED_CHAT_ID}.items(),
                        key=lambda x: x[1]["timestamp"],
                        default=(None, None)
                    )[0]
                    if recent_chat:
                        authorized_chats = {recent_chat: authorized_chats[recent_chat]}
                    else:
                        authorized_chats = {}
                    save_authorized_chats()
    except Exception as e:
        logger.error(f"Error loading authorized_chats: {e}")

# --- Check if chat_id is allowed and user is admin in group ---
async def is_allowed_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_id = str(update.effective_chat.id)
    user_id = update.effective_user.id
    current_time = time.time()
    
    if chat_id == ALLOWED_CHAT_ID:
        return True
    
    if authorized_chats and chat_id in authorized_chats:
        auth_info = authorized_chats[chat_id]
        if current_time - auth_info["timestamp"] < 24 * 3600:
            if update.effective_chat.type in ['group', 'supergroup']:
                try:
                    member = await update.effective_chat.get_member(user_id)
                    if member.status not in ['administrator', 'creator']:
                        await update.message.reply_text("Chỉ quản trị viên của nhóm mới có thể sử dụng các lệnh của bot.")
                        logger.warning(f"User {user_id} in chat {chat_id} is not an admin.")
                        return False
                except TelegramError as e:
                    logger.error(f"Error checking admin status for user {user_id} in chat {chat_id}: {e}")
                    await update.message.reply_text("Lỗi khi kiểm tra quyền quản trị viên. Vui lòng thử lại hoặc liên hệ @mekiemtienlamnha.")
                    return False
            return True
        else:
            del authorized_chats[chat_id]
            save_authorized_chats()
    
    try:
        await update.message.reply_text("Đoạn chat này không có quyền sử dụng bot. Vui lòng nhập key bằng lệnh /key <key>. Liên hệ @mekiemtienlamnha để được hỗ trợ.")
    except TelegramError as e:
        logger.error(f"Error sending unauthorized message to chat {chat_id}: {e}")
    return False

# --- Key Command ---
async def key_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    current_time = time.time()
    
    if chat_id == ALLOWED_CHAT_ID:
        await update.message.reply_text("Đoạn chat này đã được cấp quyền mặc định và không cần nhập key.")
        return
    
    if len(context.args) != 1:
        await update.message.reply_text("Vui lòng cung cấp key. Ví dụ: /key 123")
        return
    
    provided_key = context.args[0]
    try:
        valid_key = VALID_KEY
        
        if chat_id in authorized_chats:
            auth_info = authorized_chats[chat_id]
            if auth_info["key_attempts"] >= 2:
                await update.message.reply_text("Đoạn chat này đã bị khóa do nhập key quá số lần cho phép. Liên hệ @mekiemtienlamnha để được hỗ trợ.")
                logger.warning(f"Chat {chat_id} is blocked due to multiple key attempts.")
                return
            if current_time - auth_info["timestamp"] < 24 * 3600:
                await update.message.reply_text("Đoạn chat này đã được cấp quyền. Không cần nhập lại key.")
                auth_info["key_attempts"] += 1
                save_authorized_chats()
                return
            else:
                del authorized_chats[chat_id]
                save_authorized_chats()
        
        if provided_key == valid_key:
            authorized_chats.clear()
            authorized_chats[chat_id] = {"timestamp": current_time, "key_attempts": 1}
            save_authorized_chats()
            await update.message.reply_text("Key hợp lệ! Đoạn chat này đã được cấp quyền duy nhất để sử dụng bot trong 24 giờ (ngoại trừ đoạn chat mặc định).")
            logger.info(f"Chat {chat_id} authorized with key: {provided_key}")
            
            context.job_queue.run_once(
                callback=lambda ctx: remove_authorization(ctx, chat_id),
                when=24 * 3600,
                name=f"remove_auth_{chat_id}"
            )
        else:
            if chat_id not in authorized_chats:
                authorized_chats[chat_id] = {"timestamp": current_time, "key_attempts": 1}
            else:
                authorized_chats[chat_id]["key_attempts"] += 1
            save_authorized_chats()
            await update.message.reply_text("Key không hợp lệ. Vui lòng thử lại hoặc liên hệ @mekiemtienlamnha.")
            logger.warning(f"Chat {chat_id} attempted to use invalid key: {provided_key}")
    except Exception as e:
        logger.error(f"Error processing key command in chat {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi xử lý key. Vui lòng thử lại hoặc liên hệ @mekiemtienlamnha.")

# --- Remove Authorization ---
async def remove_authorization(context: ContextTypes.DEFAULT_TYPE, chat_id: str):
    if chat_id in authorized_chats:
        del authorized_chats[chat_id]
        save_authorized_chats()
        try:
            await context.bot.send_message(chat_id=chat_id, text="Quyền truy cập của đoạn chat này đã hết hạn. Vui lòng nhập lại key bằng lệnh /key <key>.")
            logger.info(f"Authorization removed for chat {chat_id} after 24 hours.")
        except TelegramError as e:
            logger.error(f"Error sending expiration message to chat {chat_id}: {e}")

# --- Get BTC Price and Volume ---
def get_btc_price_and_volume():
    global price_history, volume_history, high_history, low_history, open_history
    try:
        response = requests.get(KRAKEN_OHLC_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            logger.warning(f"Kraken API error: {data['error']}")
            return None, None, None, None, None, None, None
        
        klines = data['result']['XXBTZUSD'][:MAX_HISTORY]
        if not klines or len(klines) < MAX_HISTORY:
            logger.warning(f"Kraken data insufficient: {len(klines)} candles received")
            return None, None, None, None, None, None, None
        
        prices = [float(candle[4]) for candle in klines]
        volumes = [float(candle[5]) for candle in klines]
        highs = [float(candle[2]) for candle in klines]
        lows = [float(candle[3]) for candle in klines]
        opens = [float(candle[1]) for candle in klines]
        latest_price = prices[-1]
        latest_volume = volumes[-1]
        
        price_history = prices[-MAX_HISTORY:]
        volume_history = volumes[-MAX_HISTORY:]
        high_history = highs[-MAX_HISTORY:]
        low_history = lows[-MAX_HISTORY:]
        open_history = opens[-MAX_HISTORY:]
        
        return latest_price, latest_volume, prices, volumes, highs, lows, opens
    except requests.exceptions.HTTPError as e:
        logger.error(f"Kraken API error: {e}")
        return None, None, None, None, None, None, None
    except requests.RequestException as e:
        logger.error(f"Kraken API request error: {e}")
        return None, None, None, None, None, None, None
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Error parsing Kraken data: {e}")
        return None, None, None, None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error in get_btc_price_and_volume: {e}")
        return None, None, None, None, None, None, None

# --- Calculate VWAP ---
def calculate_vwap(prices, volumes, highs, lows, period=14):
    if len(prices) < period or len(volumes) < period:
        return None
    typical_prices = [(highs[i] + lows[i] + prices[i]) / 3 for i in range(-period, 0)]
    vwap = sum(typical_prices[i] * volumes[i] for i in range(-period, 0)) / sum(volumes[-period:])
    return vwap

# --- Calculate ATR ---
def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    tr_list = []
    for i in range(-period, 0):
        high_low = highs[i] - lows[i]
        high_prev_close = abs(highs[i] - closes[i-1])
        low_prev_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_prev_close, low_prev_close)
        tr_list.append(tr)
    return sum(tr_list) / period

# --- Calculate Fibonacci Retracement Levels ---
def calculate_fibonacci_levels(prices, period=100):
    if len(prices) < period:
        return None, None, None
    high = max(prices[-period:])
    low = min(prices[-period:])
    diff = high - low
    fib_382 = high - diff * 0.382
    fib_618 = high - diff * 0.618
    return fib_382, fib_618, diff

# --- Calculate SMA ---
def calculate_sma(prices, period=5):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

# --- Calculate EMA ---
def calculate_ema(prices, period):
    if len(prices) < period or not prices:
        return None
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema

# --- Calculate MACD ---
def calculate_macd(prices):
    if len(prices) < 26:
        return None, None
    ema_12 = calculate_ema(prices[-12:], 12)
    ema_26 = calculate_ema(prices[-26:], 26)
    if ema_12 is None or ema_26 is None:
        return None, None
    macd = ema_12 - ema_26
    macd_line = [calculate_ema(prices[max(0, i-12):i], 12) - calculate_ema(prices[max(0, i-26):i], 26) 
                 for i in range(26, len(prices))]
    if len(macd_line) < 9:
        return macd, None
    signal = calculate_ema(macd_line[-9:], 9)
    return macd, signal

# --- Calculate RSI ---
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return np.nan
    prices = [p for p in prices if p is not None]
    prices = np.array(prices, dtype=float)
    if len(prices) < period + 1:
        return np.nan
    price_changes = np.diff(prices)
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    avg_gain = np.mean(gains[-period:]) if np.sum(gains[-period:]) > 0 else 0
    avg_loss = np.mean(losses[-period:]) if np.sum(losses[-period:]) > 0 else 0
    if avg_loss == 0:
        rsi = 100 if avg_gain > 0 else 50
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Calculate Stochastic Oscillator ---
def calculate_stochastic(prices, highs, lows, k_period=14, d_period=3):
    if len(prices) < k_period or len(highs) < k_period or len(lows) < k_period:
        return None, None
    low_min = min(lows[-k_period:])
    high_max = max(highs[-k_period:])
    if high_max == low_min:
        k_value = 50
    else:
        k_value = 100 * (prices[-1] - low_min) / (high_max - low_min)
    k_values = []
    for i in range(d_period):
        if len(prices[:-i]) >= k_period:
            low_min = min(lows[-k_period-i:-i]) if lows[-k_period-i:-i] else low_min
            high_max = max(highs[-k_period-i:-i]) if highs[-k_period-i:-i] else high_max
            if high_max != low_min:
                k = 100 * (prices[-1-i] - low_min) / (high_max - low_min)
                k_values.append(k)
    d_value = np.mean([k_value] + k_values) if k_values else None
    return k_value, d_value

# --- Calculate Bollinger Bands ---
def calculate_bollinger_bands(prices, period=20, num_std=2):
    if len(prices) < period:
        return None, None, None
    sma = sum(prices[-period:]) / period
    std = np.std(prices[-period:])
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

# --- Calculate Trend Line ---
def calculate_trend_line(prices, period=10):
    if len(prices) < period:
        return None
    x = np.arange(period)
    y = np.array(prices[-period:])
    slope, _ = np.polyfit(x, y, 1)
    if slope > 0.01:
        return "Tăng - Xu hướng tăng"
    elif slope < -0.01:
        return "Giảm - Xu hướng giảm"
    else:
        return "Đi ngang - Sideways"

# --- Detect Candlestick Patterns ---
def detect_candlestick_pattern(highs, lows, opens, closes):
    if len(closes) < 3:
        return None
    body = abs(opens[-1] - closes[-1])
    range_candle = highs[-1] - lows[-1]
    prev_body = abs(opens[-2] - closes[-2])
    prev_range = highs[-2] - lows[-2]
    
    if body <= range_candle * 0.1:
        return "Doji - Tín hiệu đảo chiều tiềm năng"
    
    if prev_body > prev_range * 0.3 and body > range_candle * 0.5:
        if closes[-2] < opens[-2] and closes[-1] > opens[-1] and closes[-1] > opens[-2] and opens[-1] < closes[-2]:
            return "Bullish Engulfing - Tín hiệu tăng"
        if closes[-2] > opens[-2] and closes[-1] < opens[-1] and closes[-1] < opens[-2] and opens[-1] > closes[-2]:
            return "Bearish Engulfing - Tín hiệu giảm"
    
    lower_wick = opens[-1] - lows[-1] if opens[-1] > closes[-1] else closes[-1] - lows[-1]
    upper_wick = highs[-1] - opens[-1] if opens[-1] > closes[-1] else highs[-1] - closes[-1]
    if body < range_candle * 0.3:
        if lower_wick > body * 2 and upper_wick < body:
            return "Hammer - Tín hiệu tăng"
        if upper_wick > body * 2 and lower_wick < body:
            return "Shooting Star - Tín hiệu giảm"
    
    if len(closes) >= 3:
        if (closes[-3] < opens[-3] and abs(opens[-2] - closes[-2]) < prev_range * 0.2 and 
            closes[-1] > opens[-1] and closes[-1] > closes[-3]):
            return "Morning Star - Tín hiệu tăng"
        if (closes[-3] > opens[-3] and abs(opens[-2] - closes[-2]) < prev_range * 0.2 and 
            closes[-1] < opens[-1] and closes[-1] < closes[-3]):
            return "Evening Star - Tín hiệu giảm"
    
    if len(closes) >= 4:
        if (closes[-3] > opens[-3] and closes[-2] > opens[-2] and closes[-1] > opens[-1] and
            closes[-1] > closes[-2] > closes[-3]):
            return "Three White Soldiers - Tín hiệu tăng"
        if (closes[-3] < opens[-3] and closes[-2] < opens[-2] and closes[-1] < opens[-1] and
            closes[-1] < closes[-2] < closes[-3]):
            return "Three Black Crows - Tín hiệu giảm"
    
    return None

# --- Detect Chart Patterns ---
def detect_chart_patterns(prices, highs, lows, period=50):
    if len(prices) < period:
        return None
    highs_period = highs[-period:]
    lows_period = lows[-period:]
    
    max_highs = sorted([(h, i) for i, h in enumerate(highs_period)], reverse=True)[:2]
    min_lows = sorted([(l, i) for i, l in enumerate(lows_period)])[:2]
    if len(max_highs) >= 2 and abs(max_highs[0][0] - max_highs[1][0]) < max_highs[0][0] * 0.01 and abs(max_highs[0][1] - max_highs[1][1]) > 5:
        return "Double Top - Tín hiệu đảo chiều giảm"
    if len(min_lows) >= 2 and abs(min_lows[0][0] - min_lows[1][0]) < min_lows[0][0] * 0.01 and abs(min_lows[0][1] - min_lows[1][1]) > 5:
        return "Double Bottom - Tín hiệu đảo chiều tăng"
    
    if len(highs_period) > 10:
        peak_indices = [i for i in range(1, len(highs_period)-1) if highs_period[i] > highs_period[i-1] and highs_period[i] > highs_period[i+1]]
        if len(peak_indices) >= 3:
            left_shoulder = peak_indices[0]
            head = peak_indices[1]
            right_shoulder = peak_indices[2]
            if (highs_period[head] > highs_period[left_shoulder] and highs_period[head] > highs_period[right_shoulder] and
                abs(highs_period[left_shoulder] - highs_period[right_shoulder]) < highs_period[head] * 0.01):
                return "Head and Shoulders - Tín hiệu đảo chiều giảm"
    
    highs_trend = np.polyfit(range(len(highs_period)), highs_period, 1)[0]
    lows_trend = np.polyfit(range(len(lows_period)), lows_period, 1)[0]
    if highs_trend < 0 and lows_trend > 0 and abs(highs_trend) > 0.01 and abs(lows_trend) > 0.01:
        return "Symmetrical Triangle - Tín hiệu tích lũy"
    
    price_range = max(highs_period) - min(lows_period)
    recent_highs = highs_period[-10:]
    recent_lows = lows_period[-10:]
    if len(recent_highs) >= 10:
        flag_highs_trend = np.polyfit(range(10), recent_highs, 1)[0]
        flag_lows_trend = np.polyfit(range(10), recent_lows, 1)[0]
        if abs(flag_highs_trend) < 0.005 and abs(flag_lows_trend) < 0.005 and price_range < max(highs_period) * 0.02:
            if prices[-1] > prices[-10]:
                return "Bullish Flag - Tín hiệu tăng tiếp diễn"
            else:
                return "Bearish Flag - Tín hiệu giảm tiếp diễn"
    
    if highs_trend < 0 and lows_trend > 0 and abs(highs_trend) > abs(lows_trend) * 1.5:
        return "Rising Wedge - Tín hiệu giảm"
    elif highs_trend < 0 and lows_trend > 0 and abs(lows_trend) > abs(highs_trend) * 1.5:
        return "Falling Wedge - Tín hiệu tăng"
    
    return None

# --- Calculate Volume Spike ---
def calculate_volume_spike(volumes, period=5, threshold=1.5):
    if len(volumes) < period:
        return False
    avg_volume = np.mean(volumes[-period:])
    return volumes[-1] > avg_volume * threshold

# --- Detect Breakout ---
def detect_breakout(prices, highs, lows, volumes, period=20):
    if len(prices) < period or len(volumes) < period:
        return None, False
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]
    recent_volumes = volumes[-period:]
    dynamic_resistance = max(recent_highs)
    dynamic_support = min(recent_lows)
    latest_price = prices[-1]
    volume_spike = calculate_volume_spike(recent_volumes, period=5, threshold=2.0)
    
    if latest_price > dynamic_resistance * 1.005 and volume_spike:
        return "Breakout Up - Tín hiệu tăng mạnh", True
    elif latest_price < dynamic_support * 0.995 and volume_spike:
        return "Breakout Down - Tín hiệu giảm mạnh", True
    elif latest_price > dynamic_resistance * 1.005 or latest_price < dynamic_support * 0.995:
        return "Breakout Fake - Tín hiệu phá vỡ giả", False
    return None, False

# --- Analyze Price Behavior ---
def analyze_price_behavior(prices, highs, lows, opens, closes):
    if len(closes) < 5:
        return None
    body_sizes = [abs(opens[-i] - closes[-i]) for i in range(1, 6)]
    if all(body_sizes[i] < body_sizes[i+1] for i in range(len(body_sizes)-1)):
        return "Accumulation - Giá tích lũy"
    if closes[-1] < highs[-2] and closes[-2] < highs[-3]:
        return "Weak Uptrend - Lực mua yếu"
    if closes[-1] > lows[-2] and closes[-2] > lows[-3]:
        return "Weak Downtrend - Lực bán yếu"
    return None

# --- Check Timing Entry ---
def check_timing_entry():
    current_hour = datetime.now().hour
    economic_calendar = get_economic_calendar()
    if any("Non-Farm Payrolls" in event or "FOMC" in event for event in economic_calendar):
        return "Cảnh báo: Có tin tức lớn, tránh giao dịch!"
    if 8 <= current_hour <= 16:
        return "Good Timing - Phiên Âu/Mỹ"
    return "Caution - Tránh giờ tin tức hoặc phiên thấp điểm"

# --- Scalping Strategy ---
def scalping_strategy(prices, period_fast=5, period_slow=10):
    if len(prices) < period_slow:
        return None
    ema_fast = calculate_ema(prices, period_fast)
    ema_slow = calculate_ema(prices, period_slow)
    if ema_fast is None or ema_slow is None:
        return None
    if ema_fast > ema_slow:
        return "Scalping Buy - Tín hiệu mua nhanh"
    elif ema_fast < ema_slow:
        return "Scalping Sell - Tín hiệu bán nhanh"
    return None

# --- Predict Price with Random Forest ---
def predict_price_rf(prices, volumes, highs, lows):
    if len(prices) < 30:
        return None
    df = pd.DataFrame({
        'price': prices[-30:],
        'volume': volumes[-30:],
        'high': highs[-30:],
        'low': lows[-30:],
        'atr': [calculate_atr(highs, lows, prices) for _ in range(30)],
        'vwap': [calculate_vwap(prices, volumes, highs, lows) for _ in range(30)]
    })
    df['price'] = df['price'].ffill().fillna(0)
    df['price_diff'] = df['price'].diff()
    df['sma_5'] = df['price'].rolling(window=5).mean()
    df['rsi'] = df['price'].rolling(window=14, min_periods=14).apply(lambda x: calculate_rsi(x), raw=True)
    X = df[['price', 'volume', 'high', 'low', 'price_diff', 'sma_5', 'atr', 'vwap']].dropna()
    y = X['price'].shift(-1).dropna()
    X = X.iloc[:-1]
    if len(X) < 10:
        return None
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    next_data = X.iloc[-1:].values
    return model.predict(next_data)[0]

# --- Get Economic News ---
def get_economic_news():
    NEWS_API_URL = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(NEWS_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'articles' in data:
            return [article['title'] for article in data['articles'][:3]]
        return []
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

# --- Get Economic Calendar ---
def get_economic_calendar():
    try:
        return ["USD Non-Farm Payrolls at 14:30", "FOMC Interest Rate Decision at 20:00"]
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {e}")
        return []

# --- Calculate Risk Management ---
def calculate_risk_management(capital, risk_percentage=0.02, atr=None):
    if atr is None:
        atr = calculate_atr(high_history, low_history, price_history)
    position_size = capital * risk_percentage / atr if atr else capital * risk_percentage
    return position_size

# --- Martingale Strategy ---
def martingale_strategy(last_bet, win=False):
    return last_bet * 2 if not win else last_bet

# --- Detect Support and Resistance ---
def detect_support_resistance(price, highs, lows, period=50):
    global support_level, resistance_level
    if len(highs) < period or len(lows) < period:
        return "Không đủ dữ liệu"
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]
    support_level = min(recent_lows) * 0.99 if recent_lows else None
    resistance_level = max(recent_highs) * 1.01 if recent_highs else None
    if price is not None and support_level is not None and price <= support_level:
        return "Cảnh báo: Chạm vùng hỗ trợ mạnh!"
    elif price is not None and resistance_level is not None and price >= resistance_level:
        return "Cảnh báo: Chạm vùng kháng cự mạnh!"
    return "Ổn định"

# --- Format Value ---
def format_value(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"

# --- Save Data to CSV ---
def save_to_csv(price, trend, win_rate, market_status, chat_id):
    try:
        file_exists = os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Time', 'ChatID', 'Price', 'Trend', 'WinRate', 'MarketStatus'])
            writer.writerow([datetime.now().strftime("%H:%M:%S %d-%m-%Y"), chat_id, price, trend, win_rate, market_status])
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

# --- Get Help Text ---
def get_help_text():
    return """
Danh sách các lệnh hỗ trợ:
- /key <key>: Nhập key để cấp quyền sử dụng bot cho đoạn chat này (ví dụ: /key 123).
- /analyze: Phân tích thị trường ngay lập tức (30 giây).
- /set_up <phân_tích> <chờ kết quả> <bắt đầu> <kết thúc>: Tùy chỉnh tự động phân tích (phân tích bao nhiêu giây, chờ kết quả bao nhiêu giây, bắt đầu lúc nào, dừng lúc nào). Ví dụ: /set_up 50 70 21:30 23:30
- /stop: Dừng lệnh /set_up.
- /cskh: Liên hệ hỗ trợ qua Telegram.
- /help: Hiển thị danh sách các lệnh này.

Gõ /<lệnh> để sử dụng!
Lưu ý: Trong nhóm, chỉ quản trị viên mới có thể sử dụng các lệnh của bot.
"""

# --- Parse Time to Seconds ---
def parse_time_to_seconds(time_str):
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600 + minutes * 60
    except ValueError:
        raise ValueError("Định dạng thời gian không hợp lệ. Vui lòng dùng HH:MM (ví dụ: 21:30).")

# --- Get Current Time in Seconds ---
def current_time_in_seconds():
    now = datetime.now()
    return now.hour * 3600 + now.minute * 60 + now.second

# --- Analyze Market ---
async def analyze_market(update: Update, context: ContextTypes.DEFAULT_TYPE, analysis_duration=30):
    global price_history, volume_history, high_history, low_history, open_history, is_analyzing
    chat_id = str(update.effective_chat.id)

    if not await is_allowed_chat(update, context):
        return
    
    if is_analyzing:
        await update.message.reply_text("Phân tích đang diễn ra, vui lòng đợi.")
        return

    is_analyzing = True
    start_time = time.time()
    temp_prices = []
    temp_volumes = []
    temp_highs = []
    temp_lows = []
    temp_opens = []

    try:
        while time.time() - start_time < max(analysis_duration, MIN_ANALYSIS_DURATION):
            if context.bot_data.get('stopping', False):
                break
            latest_price, latest_volume, prices, volumes, highs, lows, opens = get_btc_price_and_volume()
            if latest_price is not None:
                temp_prices.append(latest_price)
                temp_volumes.append(latest_volume)
                temp_highs.append(highs[-1])
                temp_lows.append(lows[-1])
                temp_opens.append(opens[-1])
                price_history = prices[-MAX_HISTORY:]
                volume_history = volumes[-MAX_HISTORY:]
                high_history = highs[-MAX_HISTORY:]
                low_history = lows[-MAX_HISTORY:]
                open_history = opens[-MAX_HISTORY:]
            else:
                logger.warning("Failed to fetch data in this iteration.")
            await asyncio.sleep(1)
    finally:
        is_analyzing = False

    if not temp_prices:
        await update.message.reply_text("Không thể lấy dữ liệu từ Coincex. Vui lòng thử lại hoặc kiểm tra kết nối mạng.")
        return

    latest_price = temp_prices[-1]
    price_history = price_history[-MAX_HISTORY:]

    # Technical analysis
    sma_5 = calculate_sma(price_history, 5)
    ema_9 = calculate_ema(price_history, 9)
    ema_21 = calculate_ema(price_history, 21)
    sma_20, upper_band, lower_band = calculate_bollinger_bands(price_history)
    macd, signal = calculate_macd(price_history)
    rsi = calculate_rsi(price_history)
    k_value, d_value = calculate_stochastic(price_history, high_history, low_history)
    vwap = calculate_vwap(price_history, volume_history, high_history, low_history)
    atr = calculate_atr(high_history, low_history, price_history)
    fib_382, fib_618, fib_diff = calculate_fibonacci_levels(price_history)
    predicted_price = predict_price_rf(price_history, volume_history, high_history, low_history)
    candlestick_pattern = detect_candlestick_pattern(high_history, low_history, open_history, price_history)
    chart_pattern = detect_chart_patterns(price_history, high_history, low_history)
    trend_line = calculate_trend_line(price_history)
    volume_trend = "TĂNG" if sum(volume_history[-5:]) > sum(volume_history[-10:-5]) else "GIẢM"
    volume_spike = calculate_volume_spike(volume_history)
    breakout, is_true_breakout = detect_breakout(price_history, high_history, low_history, volume_history)
    support_resistance_signal = detect_support_resistance(latest_price, high_history, low_history)
    price_behavior = analyze_price_behavior(price_history, high_history, low_history, open_history, price_history)
    timing_entry = check_timing_entry()
    scalping_signal = scalping_strategy(price_history)

    # Data analysis
    news = get_economic_news()
    calendar = get_economic_calendar()

    # Risk management
    capital = 1000
    position_size = calculate_risk_management(capital=1000)
    last_bet = 10
    next_bet = martingale_strategy(last_bet, win=False)

    buy_signals = [
        (1.5 if macd is not None and signal is not None and macd > signal else 0, "MACD Buy"),
        (1.2 if latest_price is not None and sma_5 is not None and latest_price > sma_5 else 0, "SMA Buy"),
        (1.5 if ema_9 is not None and ema_21 is not None and ema_9 > ema_21 else 0, "EMA9 > EMA21"),
        (1.5 if volume_trend == "TĂNG" else 0, "Volume Up"),
        (1.5 if lower_band is not None and latest_price <= lower_band else 0, "Bollinger Lower"),
        (2.0 if rsi is not None and rsi < 30 else 0, "RSI Oversold"),
        (2.5 if rsi is not None and rsi < 20 else 0, "RSI Strongly Oversold"),
        (1.2 if k_value is not None and d_value is not None and k_value < 20 else 0, "Stochastic Oversold"),
        (1.5 if vwap is not None and latest_price < vwap else 0, "Below VWAP"),
        (1.3 if fib_618 is not None and latest_price <= fib_618 else 0, "Fib 61.8%"),
        (1.5 if candlestick_pattern in ["Bullish Engulfing - Tín hiệu tăng", "Hammer - Tín hiệu tăng", 
                                       "Morning Star - Tín hiệu tăng", "Three White Soldiers - Tín hiệu tăng"] else 0, candlestick_pattern),
        (1.5 if chart_pattern in ["Double Bottom - Tín hiệu đảo chiều tăng", "Bullish Flag - Tín hiệu tăng tiếp diễn", 
                                  "Falling Wedge - Tín hiệu tăng"] else 0, chart_pattern),
        (1.5 if trend_line == "Tăng - Xu hướng tăng" else 0, "Trend Up"),
        (2.0 if volume_spike else 0, "Volume Spike Buy"),
        (2.5 if breakout == "Breakout Up - Tín hiệu tăng mạnh" and is_true_breakout else 0, "Breakout Up"),
        (1.5 if price_behavior == "Accumulation - Giá tích lũy" else 0, "Price Accumulation"),
        (1.2 if timing_entry == "Good Timing - Phiên Âu/Mỹ" else 0, "Good Timing"),
        (1.5 if scalping_signal == "Scalping Buy - Tín hiệu mua nhanh" else 0, "Scalping Buy")
    ]
    
    sell_signals = [
        (1.5 if macd is not None and signal is not None and macd < signal else 0, "MACD Sell"),
        (1.2 if latest_price is not None and sma_5 is not None and latest_price < sma_5 else 0, "SMA Sell"),
        (1.5 if ema_9 is not None and ema_21 is not None and ema_9 < ema_21 else 0, "EMA9 < EMA21"),
        (1.5 if volume_trend == "GIẢM" else 0, "Volume Down"),
        (1.5 if upper_band is not None and latest_price >= upper_band else 0, "Bollinger Upper"),
        (2.0 if rsi is not None and rsi > 70 else 0, "RSI Overbought"),
        (2.5 if rsi is not None and rsi > 80 else 0, "RSI Strongly Overbought"),
        (1.2 if k_value is not None and d_value is not None and k_value > 80 else 0, "Stochastic Overbought"),
        (1.5 if vwap is not None and latest_price > vwap else 0, "Above VWAP"),
        (1.3 if fib_382 is not None and latest_price >= fib_382 else 0, "Fib 38.2%"),
        (1.5 if candlestick_pattern in ["Bearish Engulfing - Tín hiệu giảm", "Shooting Star - Tín hiệu giảm", 
                                       "Evening Star - Tín hiệu giảm", "Three Black Crows - Tín hiệu giảm"] else 0, candlestick_pattern),
        (1.5 if chart_pattern in ["Double Top - Tín hiệu đảo chiều giảm", "Head and Shoulders - Tín hiệu đảo chiều giảm", 
                                  "Rising Wedge - Tín hiệu giảm", "Bearish Flag - Tín hiệu giảm tiếp diễn"] else 0, chart_pattern),
        (1.5 if trend_line == "Giảm - Xu hướng giảm" else 0, "Trend Down"),
        (2.0 if volume_spike else 0, "Volume Spike Sell"),
        (2.5 if breakout == "Breakout Down - Tín hiệu giảm mạnh" and is_true_breakout else 0, "Breakout Down"),
        (1.5 if price_behavior == "Weak Uptrend - Lực mua yếu" else 0, "Weak Uptrend"),
        (1.0 if timing_entry == "Caution - Tránh giờ tin tức hoặc phiên thấp điểm" else 0, "Poor Timing"),
        (1.5 if scalping_signal == "Scalping Sell - Tín hiệu bán nhanh" else 0, "Scalping Sell")
    ]

    buy_score = sum(weight for weight, _ in buy_signals)
    sell_score = sum(weight for weight, _ in sell_signals)

    trend = "MUA" if buy_score > sell_score else "BÁN" if sell_score > buy_score else "CHỜ LỆNH"
    total_score = buy_score + sell_score
    win_rate = ((buy_score / total_score) * 100 if trend == "MUA" else (sell_score / total_score) * 100 if trend == "BÁN" else 50) if total_score > 0 else 50

    latest_price_str = format_value(latest_price)
    win_rate_str = format_value(win_rate)
    market_status = f"{support_resistance_signal}"

    report = f"""
📈 **COINCEX - BTC/USD** 📈
🕒 **Thời gian**: {datetime.now().strftime('%H:%M:%S %d-%m-%Y')}
💹 **Giá hiện tại**: {latest_price_str} USD
🚨 **Tín hiệu**: {trend}
🎯 **Tỷ lệ thắng**: {win_rate_str}%
📋 **Trạng thái**: {market_status}
"""

    try:
        await update.message.reply_text(report)
        logger.info(f"Analysis report sent to chat {chat_id}")
        save_to_csv(latest_price, trend, win_rate, market_status, chat_id)
    except TelegramError as e:
        logger.error(f"Error sending Telegram message to {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi gửi báo cáo. Vui lòng thử lại.")

# --- Analyze Command ---
async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await analyze_market(update, context, analysis_duration=30)

# --- Set Up Callback ---
async def set_up_callback(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data['chat_id']
    update = context.job.data['update']
    analysis_duration = context.job.data['analysis_duration']
    await analyze_market(update, context, analysis_duration)

# --- Send Start Trade Message ---
async def send_start_trade_message(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data['chat_id']
    try:
        await context.bot.send_message(chat_id=chat_id, text="BẮT ĐẦU TRADE NÀO")
    except TelegramError as e:
        logger.error(f"Error sending start trade message to {chat_id}: {e}")

# --- Set Up Command ---
async def set_up_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_allowed_chat(update, context):
        return
    chat_id = str(update.effective_chat.id)
    if chat_id in set_up_jobs:
        await update.message.reply_text("Lệnh /set_up đã đang chạy. Dùng /stop để dừng trước khi chạy lại.")
        return

    if len(context.args) != 4:
        await update.message.reply_text("Vui lòng cung cấp đúng định dạng: /set_up <phân_tích> <chờ kết quả> <bắt đầu> <kết thúc>\nVí dụ: /set_up 50 70 21:30 23:30")
        return

    try:
        analysis_duration = int(context.args[0])
        rest_duration = int(context.args[1])
        start_time = context.args[2]
        end_time = context.args[3]

        if analysis_duration < MIN_ANALYSIS_DURATION:
            await update.message.reply_text(f"Thời gian phân tích phải lớn hơn hoặc bằng {MIN_ANALYSIS_DURATION} giây.")
            return
        if rest_duration < MIN_REST_DURATION:
            await update.message.reply_text(f"Thời gian chờ kết quả phải lớn hơn hoặc bằng {MIN_REST_DURATION} giây.")
            return

        start_seconds = parse_time_to_seconds(start_time)
        end_seconds = parse_time_to_seconds(end_time)
        current_seconds = current_time_in_seconds()

        if end_seconds <= start_seconds:
            end_seconds += 24 * 3600
        delay = start_seconds - current_seconds
        if delay < 0:
            delay += 24 * 3600
        interval = analysis_duration + rest_duration
        duration = end_seconds - start_seconds

        await update.message.reply_text(f"Bắt đầu tự động phân tích từ {start_time} đến {end_time}, phân tích {analysis_duration} giây, chờ kết quả {rest_duration} giây mỗi chu kỳ. Dùng /stop để dừng.")

        if delay >= 10:
            context.job_queue.run_once(
                callback=send_start_trade_message,
                when=delay - 10,
                data={'chat_id': chat_id},
                name=f"start_trade_message_{chat_id}"
            )
        else:
            await context.bot.send_message(chat_id=chat_id, text="BẮT ĐẦU TRADE NÀO")

        job = context.job_queue.run_repeating(
            callback=set_up_callback,
            interval=interval,
            first=delay,
            data={'chat_id': chat_id, 'update': update, 'analysis_duration': analysis_duration},
            name=f"set_up_{chat_id}"
        )

        context.job_queue.run_once(
            callback=lambda ctx: stop_set_up(ctx, chat_id),
            when=delay + duration,
            name=f"stop_set_up_{chat_id}"
        )

        set_up_jobs[chat_id] = job

    except ValueError as e:
        await update.message.reply_text(f"Lỗi: {str(e)}. Vui lòng kiểm tra định dạng thời gian hoặc số liệu.")
    except Exception as e:
        await update.message.reply_text(f"Lỗi không xác định: {str(e)}")

# --- Stop Command ---
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_allowed_chat(update, context):
        return
    chat_id = str(update.message.chat.id)
    stopped = False
    if chat_id in set_up_jobs:
        await stop_set_up(context, chat_id)
        stopped = True
    if not stopped:
        await update.message.reply_text("Không có lệnh tự động nào đang chạy (/set_up).")

# --- CSKH Command ---
async def cskh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_allowed_chat(update, context):
        return
    support_message = "Cần hỗ trợ? Liên hệ qua Telegram: @mekiemtienlamnha"
    await update.message.reply_text(support_message)

# --- Help Command ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_allowed_chat(update, context):
        return
    help_text = get_help_text()
    await update.message.reply_text(help_text)

# --- Stop Set Up ---
async def stop_set_up(context: ContextTypes.DEFAULT_TYPE, chat_id: str):
    if chat_id in set_up_jobs:
        set_up_jobs[chat_id].schedule_removal()
        del set_up_jobs[chat_id]
        try:
            await context.bot.send_message(chat_id=chat_id, text="Đã dừng tự động phân tích (/set_up).")
        except TelegramError as e:
            logger.error(f"Error sending stop message to {chat_id}: {e}")

# --- Start Bot ---
async def start_bot(app: Application):
    try:
        logger.info(f"Bot started successfully. Allowed chat_id: {ALLOWED_CHAT_ID}")
        await app.initialize()
        await app.updater.initialize()
        await app.updater.start_polling()
        await app.start()
        await asyncio.Event().wait()
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise
    finally:
        await app.stop()

# --- Run FastAPI and Telegram Bot ---
def run_bot():
    try:
        print("Loading authorized chats...")
        load_authorized_chats()
        logger.info(f"Loaded authorized_chats: {authorized_chats}")
        
        print("Initializing Telegram bot...")
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        logger.info(f"Bot initialized with TELEGRAM_TOKEN: {TELEGRAM_TOKEN}")

        print("Adding command handlers...")
        application.add_handler(CommandHandler("key", key_command))
        application.add_handler(CommandHandler("analyze", analyze_command))
        application.add_handler(CommandHandler("set_up", set_up_command))
        application.add_handler(CommandHandler("stop", stop_command))
        application.add_handler(CommandHandler("cskh", cskh_command))
        application.add_handler(CommandHandler("help", help_command))
        print("Handlers added.")

        asyncio.run(start_bot(application))
    except Exception as e:
        logger.error(f"Failed to run bot: {e}")

if __name__ == "__main__":
    # Start Telegram bot in a separate thread
    bot_thread = Thread(target=run_bot)
    bot_thread.start()

    # Run FastAPI server for Render
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
