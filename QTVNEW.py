import asyncio
import requests
import json
import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes, ConversationHandler, MessageHandler
from telegram.ext import filters
from telegram.error import TelegramError
import time
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

# --- Setup Logging ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Configuration from Environment Variables ---
CONFIG_FILE = "/opt/render/project/data/config.json"
AUTHORIZED_CHATS_FILE = "/opt/render/project/data/authorized_chats.json"
HISTORY_FILE = "/opt/render/project/data/price_history.csv"
MAX_HISTORY = 100

# Đọc cấu hình từ biến môi trường
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7608384401:AAHKfX5KlBl5CZTaoKSDwwdATmbY8Z34vRk")
ALLOWED_CHAT_ID = os.getenv("ALLOWED_CHAT_ID", "-1002554202438")
VALID_KEY = os.getenv("VALID_KEY", "10092006")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "af9b016f3f044a6f84453bbe1a526f0b")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "2b9f4c81-6f2a-4e3b-9d1e-123456789abc")

# Kiểm tra các giá trị bắt buộc
if not TELEGRAM_TOKEN or not ALLOWED_CHAT_ID:
    logger.error("TELEGRAM_TOKEN hoặc ALLOWED_CHAT_ID bị thiếu trong biến môi trường.")
    raise ValueError("TELEGRAM_TOKEN hoặc ALLOWED_CHAT_ID bị thiếu trong biến môi trường.")

# --- Constants ---
KRAKEN_OHLC_URL = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1'

# --- Global Variables ---
price_history = []
volume_history = []
high_history = []
low_history = []
open_history = []
support_level = None
resistance_level = None
authorized_chats = {}
is_analyzing = False

# --- Save and Load Authorized Chats ---
def save_authorized_chats():
    try:
        os.makedirs(os.path.dirname(AUTHORIZED_CHATS_FILE), exist_ok=True)
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
                        {k: v for k, v in authorized_chats.items() if k != ALLOWED_CHAT_ID}.items(),
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
        authorized_chats = {}

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
                    await update.message.reply_text("Lỗi khi kiểm tra quyền quản trị viên. Vui lòng thử lại.")
                    return False
            return True
        else:
            del authorized_chats[chat_id]
            save_authorized_chats()
    
    await update.message.reply_text("Đoạn chat này không có quyền sử dụng bot. Vui lòng nhập key bằng lệnh /key <key>.")
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
    if provided_key == VALID_KEY:
        authorized_chats.clear()
        authorized_chats[chat_id] = {"timestamp": current_time, "key_attempts": 1}
        save_authorized_chats()
        await update.message.reply_text("Key hợp lệ! Đoạn chat này đã được cấp quyền duy nhất trong 24 giờ.")
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
        await update.message.reply_text("Key không hợp lệ. Vui lòng thử lại.")
        logger.warning(f"Chat {chat_id} attempted to use invalid key: {provided_key}")

async def remove_authorization(context: ContextTypes.DEFAULT_TYPE, chat_id: str):
    if chat_id in authorized_chats:
        del authorized_chats[chat_id]
        save_authorized_chats()
        try:
            await context.bot.send_message(chat_id=chat_id, text="Quyền truy cập của đoạn chat này đã hết hạn. Vui lòng nhập lại key bằng lệnh /key <key>.")
            logger.info(f"Authorization removed for chat {chat_id} after 24 hours.")
        except TelegramError as e:
            logger.error(f"Error sending expiration message to chat {chat_id}: {e}")

# --- Get BTC Price and Volume with Retry ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), before_sleep=lambda retry_state: logger.warning(f"Retrying Kraken API call {retry_state.attempt_number}/3..."))
def get_btc_price_and_volume():
    global price_history, volume_history, high_history, low_history, open_history
    try:
        response = requests.get(KRAKEN_OHLC_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            logger.warning(f"Kraken API error: {data['error']}")
            return None, None, None, None, None, None, None
        
        result = data.get('result', {})
        klines = result.get('XXBTZUSD', [])
        if not klines or len(klines) < 1:
            logger.warning(f"Kraken data insufficient or empty: {len(klines)} candles received")
            return None, None, None, None, None, None, None
        
        prices = [float(candle[4]) for candle in klines[-MAX_HISTORY:]]
        volumes = [float(candle[5]) for candle in klines[-MAX_HISTORY:]]
        highs = [float(candle[2]) for candle in klines[-MAX_HISTORY:]]
        lows = [float(candle[3]) for candle in klines[-MAX_HISTORY:]]
        opens = [float(candle[1]) for candle in klines[-MAX_HISTORY:]]
        latest_price = prices[-1] if prices else None
        latest_volume = volumes[-1] if volumes else None
        
        price_history = prices[-MAX_HISTORY:]
        volume_history = volumes[-MAX_HISTORY:]
        high_history = highs[-MAX_HISTORY:]
        low_history = lows[-MAX_HISTORY:]
        open_history = opens[-MAX_HISTORY:]
        
        return latest_price, latest_volume, prices, volumes, highs, lows, opens
    except requests.exceptions.RequestException as e:
        logger.error(f"Kraken API request failed: {e}")
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
    try:
        typical_prices = [(highs[i] + lows[i] + prices[i]) / 3 for i in range(-period, 0)]
        vwap = sum(typical_prices[i] * volumes[i] for i in range(-period, 0)) / sum(volumes[-period:])
        return vwap
    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        return None

# --- Calculate ATR ---
def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    try:
        tr_list = []
        for i in range(-period, 0):
            high_low = highs[i] - lows[i]
            high_prev_close = abs(highs[i] - closes[i-1])
            low_prev_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_prev_close, low_prev_close)
            tr_list.append(tr)
        return sum(tr_list) / period
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None

# --- Calculate Fibonacci Retracement Levels ---
def calculate_fibonacci_levels(prices, period=100):
    if len(prices) < period:
        return None, None, None
    try:
        high = max(prices[-period:])
        low = min(prices[-period:])
        diff = high - low
        fib_382 = high - diff * 0.382
        fib_618 = high - diff * 0.618
        return fib_382, fib_618, diff
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return None, None, None

# --- Calculate SMA ---
def calculate_sma(prices, period=5):
    if len(prices) < period:
        return None
    try:
        return sum(prices[-period:]) / period
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return None

# --- Calculate EMA ---
def calculate_ema(prices, period):
    if not prices or len(prices) < period:
        return None
    try:
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return None

# --- Calculate MACD ---
def calculate_macd(prices):
    if len(prices) < 26:
        return None, None
    try:
        ema_12 = calculate_ema(prices[-12:], 12)
        ema_26 = calculate_ema(prices[-26:], 26)
        if ema_12 is None or ema_26 is None:
            return None, None
        macd = ema_12 - ema_26
        macd_line = [
            calculate_ema(prices[max(0, i-12):i], 12) - calculate_ema(prices[max(0, i-26):i], 26)
            for i in range(26, len(prices))
        ]
        if len(macd_line) < 9:
            return macd, None
        signal = calculate_ema(macd_line[-9:], 9)
        return macd, signal
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return None, None

# --- Calculate RSI ---
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return np.nan
    try:
        prices = [p for p in prices if p is not None]
        if len(prices) < period + 1:
            return np.nan
        prices = np.array(prices, dtype=float)
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
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return np.nan

# --- Calculate Stochastic Oscillator ---
def calculate_stochastic(prices, highs, lows, k_period=14, d_period=3):
    if len(prices) < k_period or len(highs) < k_period or len(lows) < k_period:
        return None, None
    try:
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
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        return None, None

# --- Calculate Bollinger Bands ---
def calculate_bollinger_bands(prices, period=20, num_std=2):
    if len(prices) < period:
        return None, None, None
    try:
        sma = sum(prices[-period:]) / period
        std = np.std(prices[-period:])
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        return sma, upper_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return None, None, None

# --- Calculate ADX ---
def calculate_adx(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    try:
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ])
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                 np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                  np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        df['tr_smooth'] = df['tr'].rolling(window=period).mean()
        df['dm_plus_smooth'] = df['dm_plus'].rolling(window=period).mean()
        df['dm_minus_smooth'] = df['dm_minus'].rolling(window=period).mean()
        df['di_plus'] = (df['dm_plus_smooth'] / df['tr_smooth']) * 100
        df['di_minus'] = (df['dm_minus_smooth'] / df['tr_smooth']) * 100
        df['dx'] = (abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])) * 100
        adx = df['dx'].rolling(window=period).mean().iloc[-1]
        return adx if not np.isnan(adx) else None
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        return None

# --- Calculate MFI ---
def calculate_mfi(highs, lows, closes, volumes, period=14):
    if len(closes) < period + 1:
        return None
    try:
        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(-period-1, 0)]
        raw_money_flow = [typical_prices[i] * volumes[i] for i in range(-period-1, 0)]
        positive_flow = [rmf if typical_prices[i] > typical_prices[i-1] else 0 for i, rmf in enumerate(raw_money_flow[1:], 1)]
        negative_flow = [rmf if typical_prices[i] < typical_prices[i-1] else 0 for i, rmf in enumerate(raw_money_flow[1:], 1)]
        positive_sum = sum(positive_flow)
        negative_sum = sum(negative_flow)
        if negative_sum == 0:
            return 100 if positive_sum > 0 else 50
        money_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    except Exception as e:
        logger.error(f"Error calculating MFI: {e}")
        return None

# --- Fetch On-Chain Data ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_onchain_data():
    try:
        url = "https://api.glassnode.com/v1/metrics/addresses/active_count"
        params = {"a": "BTC", "api_key": GLASSNODE_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        active_addresses = [d['v'] for d in data][-1] if data else 0
        return active_addresses
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching on-chain data: {e}")
        return 0
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Error parsing on-chain data: {e}")
        return 0

# --- Detect Candlestick Patterns ---
def detect_candlestick_pattern(highs, lows, opens, closes):
    if len(closes) < 2:
        return None
    try:
        if abs(opens[-1] - closes[-1]) <= (highs[-1] - lows[-1]) * 0.1:
            return "Doji - Tín hiệu đảo chiều tiềm năng"
        return None
    except Exception as e:
        logger.error(f"Error detecting candlestick pattern: {e}")
        return None

# --- Calculate Volume Spike ---
def calculate_volume_spike(volumes, period=5, threshold=1.5):
    if len(volumes) < period:
        return False
    try:
        avg_volume = np.mean(volumes[-period:])
        return volumes[-1] > avg_volume * threshold
    except Exception as e:
        logger.error(f"Error calculating volume spike: {e}")
        return False

# --- Detect Breakout ---
def detect_breakout(prices, highs, lows, period=20):
    if len(prices) < period:
        return None
    try:
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        dynamic_resistance = max(recent_highs)
        dynamic_support = min(recent_lows)
        latest_price = prices[-1]
        if latest_price > dynamic_resistance * 1.005:
            return "Breakout Up"
        elif latest_price < dynamic_support * 0.995:
            return "Breakout Down"
        return None
    except Exception as e:
        logger.error(f"Error detecting breakout: {e}")
        return None

# --- Predict Price with Random Forest ---
def predict_price_rf(prices, volumes, highs, lows):
    if len(prices) < 30:
        return None
    try:
        atr_value = calculate_atr(highs, lows, prices)
        vwap_value = calculate_vwap(prices, volumes, highs, lows)
        df = pd.DataFrame({
            'price': prices[-30:],
            'volume': volumes[-30:],
            'high': highs[-30:],
            'low': lows[-30:]
        })
        df['price'] = df['price'].ffill().fillna(0)
        df['price_diff'] = df['price'].diff()
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['atr'] = [atr_value] * 30 if atr_value is not None else [0] * 30
        df['vwap'] = [vwap_value] * 30 if vwap_value is not None else [0] * 30
        X = df[['price', 'volume', 'high', 'low', 'price_diff', 'sma_5', 'atr', 'vwap']].dropna()
        y = X['price'].shift(-1).dropna()
        X = X.iloc[:-1]
        if len(X) < 10:
            return None
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        logger.info("Training Random Forest model...")
        model.fit(X, y)
        next_data = X.iloc[-1:].values
        return model.predict(next_data)[0]
    except Exception as e:
        logger.error(f"Error predicting price with Random Forest: {e}")
        return None

# --- Predict Price with Linear Regression ---
def predict_price_linear(prices, volumes, highs, lows):
    if len(prices) < 30:
        return None
    try:
        scaler = MinMaxScaler()
        df = pd.DataFrame({
            'price': prices[-30:],
            'volume': volumes[-30:],
            'high': highs[-30:],
            'low': lows[-30:]
        })
        df = df.ffill().fillna(0)
        scaled_data = scaler.fit_transform(df)
        X = scaled_data[:-1]
        y = scaled_data[1:, 0]
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([scaled_data[-1]])
        return scaler.inverse_transform([[prediction[0], 0, 0, 0]])[0][0]
    except Exception as e:
        logger.error(f"Error predicting price with Linear Regression: {e}")
        return None

# --- Detect Support and Resistance ---
def detect_support_resistance(price):
    global support_level, resistance_level
    try:
        if support_level is None or (price is not None and price < support_level * 0.98):
            support_level = price * 0.98 if price is not None else None
        if resistance_level is None or (price is not None and price > resistance_level * 1.02):
            resistance_level = price * 1.02 if price is not None else None
        if price is not None and support_level is not None and price <= support_level:
            return "Cảnh báo: Chạm vùng hỗ trợ mạnh!"
        elif price is not None and resistance_level is not None and price >= resistance_level:
            return "Cảnh báo: Chạm vùng kháng cự mạnh!"
        return "Ổn định"
    except Exception as e:
        logger.error(f"Error detecting support/resistance: {e}")
        return "Ổn định"

# --- Format Value ---
def format_value(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{value:.{decimals}f}"
    except Exception as e:
        logger.error(f"Error formatting value: {e}")
        return "N/A"

# --- Save Data to CSV ---
def save_to_csv(price, trend, win_rate, market_status, chat_id):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        file_exists = os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Time', 'ChatID', 'Price', 'Trend', 'WinRate', 'MarketStatus'])
            writer.writerow([datetime.now().strftime("%H:%M:%S %d-%m-%Y"), chat_id, price, trend, win_rate, market_status])
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

# --- Start Command ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"Chào {user.first_name}! Chào mừng bạn đến với @mekiemtien102\n\n"
        "Sử dụng /help để xem danh sách lệnh.\n"
        "Nếu bạn cần hỗ trợ, hãy dùng /cskh."
    )

# --- Set Up Command ---
async def set_up(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    current_time = time.time()
    
    auth_status = "Đã được ủy quyền mặc định" if chat_id == ALLOWED_CHAT_ID else "Chưa được ủy quyền"
    if chat_id in authorized_chats:
        auth_info = authorized_chats[chat_id]
        if current_time - auth_info["timestamp"] < 24 * 3600:
            auth_status = "Đã được ủy quyền (hết hạn sau {:.1f} giờ)".format((24 * 3600 - (current_time - auth_info["timestamp"])) / 3600)
    
    api_status = "Chưa cài đặt App Account ID"
    if context.user_data.get('app_account_id'):
        api_status = f"App Account ID: {context.user_data['app_account_id']}"
    
    message = (
        f"📋 **Trạng thái thiết lập bot** 📋\n\n"
        f"**Chat ID**: {chat_id}\n"
        f"**Trạng thái ủy quyền**: {auth_status}\n"
        f"**Cài đặt API**:\n{api_status}\n\n"
        f"Nếu chưa ủy quyền, dùng /key <key>.\n"
        f"Để cài đặt App Account ID, dùng /settings."
    )
    await update.message.reply_text(message)

# --- Help Command ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_message = (
        "📖 **Hướng dẫn sử dụng @mekiemtien102** 📖\n\n"
        "Danh sách lệnh khả dụng:\n"
        "/start - Khởi động bot và xem giới thiệu\n"
        "/key <key> - Xác thực quyền truy cập cho chat (24 giờ)\n"
        "/signals - Xem tín hiệu giao dịch BTC/USD\n"
        "/settings - Cài đặt App Account ID của Coincex\n"
        "/set_up - Kiểm tra trạng thái thiết lập của bot\n"
        "/help - Hiển thị danh sách lệnh (bạn đang xem)\n"
        "/cskh - Thông tin hỗ trợ khách hàng\n\n"
        "Bot sẽ tự động lưu lịch sử tín hiệu vào file price_history.csv."
    )
    await update.message.reply_text(help_message)

# --- CSKH Command ---
async def cskh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cskh_message = (
        "📞 **Hỗ trợ khách hàng @mekiemtien102** 📞\n\n"
        "Nếu bạn cần hỗ trợ, vui lòng liên hệ:\n"
        "- **Telegram**: @mekiemtien102\n"
        "- **Email**: nguyenvietminhquang981@gmail.com\n"
        "- **Hotline**: +84 989916741 (9:00 - 17:00 GMT+7)\n\n"
        "Chúng tôi luôn sẵn sàng giúp bạn!"
    )
    await update.message.reply_text(cskh_message)

# --- Signals Command ---
async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global price_history, volume_history, high_history, low_history, open_history, is_analyzing
    chat_id = str(update.effective_chat.id)

    if not await is_allowed_chat(update, context):
        return
    
    if is_analyzing:
        await update.message.reply_text("Phân tích đang diễn ra, vui lòng đợi.")
        return

    is_analyzing = True
    try:
        latest_price, latest_volume, prices, volumes, highs, lows, opens = get_btc_price_and_volume()
        if latest_price is None:
            await update.message.reply_text("Không thể lấy dữ liệu giá. Vui lòng thử lại hoặc kiểm tra kết nối.")
            return

        price_history = prices[-MAX_HISTORY:]
        volume_history = volumes[-MAX_HISTORY:]
        high_history = highs[-MAX_HISTORY:]
        low_history = lows[-MAX_HISTORY:]
        open_history = opens[-MAX_HISTORY:]

        if not price_history or not volume_history or not high_history or not low_history or not open_history:
            await update.message.reply_text("Dữ liệu không đủ để phân tích. Vui lòng thử lại.")
            return

        # Technical analysis
        sma_5 = calculate_sma(price_history, 5)
        sma_20, upper_band, lower_band = calculate_bollinger_bands(price_history)
        macd, signal = calculate_macd(price_history)
        rsi = calculate_rsi(price_history)
        k_value, d_value = calculate_stochastic(price_history, high_history, low_history)
        vwap = calculate_vwap(price_history, volume_history, high_history, low_history)
        atr = calculate_atr(high_history, low_history, price_history)
        fib_382, fib_618, fib_diff = calculate_fibonacci_levels(price_history)
        predicted_price = predict_price_rf(price_history, volume_history, high_history, low_history)
        linear_predicted_price = predict_price_linear(price_history, volume_history, high_history, low_history)
        candlestick_pattern = detect_candlestick_pattern(high_history, low_history, open_history, price_history)
        volume_trend = "TĂNG" if sum(volume_history[-5:]) > sum(volume_history[-10:-5]) else "GIẢM"
        volume_spike = calculate_volume_spike(volume_history)
        breakout = detect_breakout(price_history, high_history, low_history)
        support_resistance_signal = detect_support_resistance(latest_price)
        adx = calculate_adx(high_history, low_history, price_history)
        mfi = calculate_mfi(high_history, low_history, price_history, volume_history)
        active_addresses = fetch_onchain_data()

        onchain_signal = "Bullish" if active_addresses > 1000000 else "Bearish" if active_addresses < 500000 else "Neutral"
        trend_strength = "Mạnh" if adx is not None and adx > 25 else "Yếu" if adx is not None else "N/A"

        buy_signals = [
            (1.5 if macd is not None and signal is not None and macd > signal else 0, "MACD Buy"),
            (1.2 if latest_price is not None and sma_5 is not None and latest_price > sma_5 else 0, "SMA Buy"),
            (1.5 if volume_trend == "TĂNG" else 0, "Volume Up"),
            (1.5 if lower_band is not None and latest_price <= lower_band else 0, "Bollinger Lower"),
            (2.0 if rsi is not None and not np.isnan(rsi) and rsi < 30 else 0, "RSI Oversold"),
            (2.5 if rsi is not None and not np.isnan(rsi) and rsi < 20 else 0, "RSI Strongly Oversold"),
            (1.2 if k_value is not None and d_value is not None and k_value < 20 else 0, "Stochastic Oversold"),
            (1.5 if vwap is not None and latest_price < vwap else 0, "Below VWAP"),
            (1.3 if fib_618 is not None and latest_price <= fib_618 else 0, "Fib 61.8%"),
            (1.5 if candlestick_pattern == "Doji - Tín hiệu đảo chiều tiềm năng" else 0, "Doji Buy"),
            (2.0 if volume_spike else 0, "Volume Spike Buy"),
            (2.5 if breakout == "Breakout Up" else 0, "Breakout Up"),
            (1.5 if adx is not None and adx > 25 and latest_price > sma_5 else 0, "Strong Uptrend (ADX)"),
            (1.5 if linear_predicted_price is not None and linear_predicted_price > latest_price else 0, "Linear Buy"),
            (1.5 if onchain_signal == "Bullish" else 0, "On-Chain Bullish"),
            (1.5 if mfi is not None and mfi < 20 else 0, "MFI Oversold")
        ]    
        sell_signals = [
            (1.5 if macd is not None and signal is not None and macd < signal else 0, "MACD Sell"),
            (1.2 if latest_price is not None and sma_5 is not None and latest_price < sma_5 else 0, "SMA Sell"),
            (1.5 if volume_trend == "GIẢM" else 0, "Volume Down"),
            (1.5 if upper_band is not None and latest_price >= upper_band else 0, "Bollinger Upper"),
            (2.0 if rsi is not None and not np.isnan(rsi) and rsi > 70 else 0, "RSI Overbought"),
            (2.5 if rsi is not None and not np.isnan(rsi) and rsi > 80 else 0, "RSI Strongly Overbought"),
            (1.2 if k_value is not None and d_value is not None and k_value > 80 else 0, "Stochastic Overbought"),
            (1.5 if vwap is not None and latest_price > vwap else 0, "Above VWAP"),
            (1.3 if fib_382 is not None and latest_price >= fib_382 else 0, "Fib 38.2%"),
            (1.5 if candlestick_pattern == "Doji - Tín hiệu đảo chiều tiềm năng" else 0, "Doji Sell"),
            (2.0 if volume_spike else 0, "Volume Spike Sell"),
            (2.5 if breakout == "Breakout Down" else 0, "Breakout Down"),
            (1.5 if adx is not None and adx > 25 and latest_price < sma_5 else 0, "Strong Downtrend (ADX)"),
            (1.5 if linear_predicted_price is not None and linear_predicted_price < latest_price else 0, "Linear Sell"),
            (1.5 if onchain_signal == "Bearish" else 0, "On-Chain Bearish"),
            (1.5 if mfi is not None and mfi > 80 else 0, "MFI Overbought")
        ]

        buy_score = sum(weight for weight, _ in buy_signals)
        sell_score = sum(weight for weight, _ in sell_signals)

        trend = "MUA" if buy_score > sell_score else "BÁN" if sell_score > buy_score else "CHỜ LỆNH"
        total_score = buy_score + sell_score
        win_rate = ((buy_score / total_score) * 100 if trend == "MUA" else (sell_score / total_score) * 100 if trend == "BÁN" else 50) if total_score > 0 else 50

        latest_price_str = format_value(latest_price)
        win_rate_str = format_value(win_rate)

        report = f"""
🏗️ COINCEX — BTC/USD 🌐
Time: {datetime.now().strftime('%H:%M:%S %d-%m-%Y')}
Lệnh: {trend}
Tỷ lệ thắng: {win_rate_str}%
Giá hiện tại: {latest_price_str} USD
"""

        try:
            await update.message.reply_text(report)
            logger.info(f"Signals report sent to chat {chat_id}")
            save_to_csv(latest_price, trend, win_rate, support_resistance_signal, chat_id)
        except TelegramError as e:
            logger.error(f"Error sending Telegram message to {chat_id}: {e}")
            await update.message.reply_text("Lỗi khi gửi tín hiệu. Vui lòng thử lại.")
    except Exception as e:
        logger.error(f"Error in signals command: {e}")
        await update.message.reply_text("Lỗi khi xử lý tín hiệu. Vui lòng thử lại.")
    finally:
        is_analyzing = False

# --- Settings Command ---
APP_ACCOUNT_ID = range(1)

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_allowed_chat(update, context):
        return
    await update.message.reply_text("Vui lòng nhập App Account ID của bạn từ Coincex:")
    return APP_ACCOUNT_ID

async def get_app_account_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['app_account_id'] = update.message.text
    await update.message.reply_text("Cài đặt App Account ID thành công! Sử dụng /signals để xem tín hiệu.")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Đã hủy cài đặt.")
    return ConversationHandler.END

# --- Validate Telegram Token ---
async def validate_token(token: str) -> bool:
    try:
        bot = Bot(token)
        await bot.get_me()
        return True
    except TelegramError as e:
        logger.error(f"Invalid Telegram token: {e}")
        return False

# --- Main Function ---
async def main():
    # Validate token before starting bot
    if not await validate_token(TELEGRAM_TOKEN):
        logger.error("Bot cannot start due to invalid Telegram token.")
        return

    print("Loading authorized chats...")
    load_authorized_chats()
    print("Initializing Telegram bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    print("Bot initialized. Adding handlers...")
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("set_up", set_up))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cskh", cskh))
    application.add_handler(CommandHandler("signals", signals))
    application.add_handler(CommandHandler("key", key_command))
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("settings", settings)],
        states={
            APP_ACCOUNT_ID: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_app_account_id)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    ))
    print("Handlers added. Starting polling...")

    # Initialize and start the bot
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    # Keep the bot running until stopped
    try:
        stop_event = asyncio.Event()
        await stop_event.wait()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping bot...")
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Bot shutdown complete.")

# --- Run the bot ---
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Failed to run bot: {e}")
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        loop.stop()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
