import asyncio
import json
import os
import csv
from datetime import datetime
import time
import logging
import aiohttp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
from dotenv import load_dotenv
from key_manager import is_key_valid

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Configuration from Environment ---
CONFIG_FILE = "api_config.json"
AUTHORIZED_CHATS_FILE = "authorized_chats.json"

def create_default_config():
    default_config = {
        "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", ""),
        "ALLOWED_CHAT_ID": os.getenv("ALLOWED_CHAT_ID", ""),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY", ""),
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        logger.warning("Created default api_config.json with environment values.")
        return default_config
    except Exception as e:
        logger.error(f"Error creating default api_config.json: {e}")
        return default_config

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if not isinstance(config, dict):
                logger.error(f"{CONFIG_FILE} is not a valid JSON object")
                raise ValueError("Invalid config format")
            TELEGRAM_TOKEN = config.get('TELEGRAM_TOKEN', os.getenv("TELEGRAM_TOKEN", ""))
            ALLOWED_CHAT_ID = config.get('ALLOWED_CHAT_ID', os.getenv("ALLOWED_CHAT_ID", ""))
            NEWS_API_KEY = config.get('NEWS_API_KEY', os.getenv("NEWS_API_KEY", ""))
            if not TELEGRAM_TOKEN or not ALLOWED_CHAT_ID:
                raise ValueError("TELEGRAM_TOKEN or ALLOWED_CHAT_ID missing in api_config.json or environment")
            return TELEGRAM_TOKEN, ALLOWED_CHAT_ID, NEWS_API_KEY
        else:
            config = create_default_config()
            return config.get('TELEGRAM_TOKEN'), config.get('ALLOWED_CHAT_ID'), config.get('NEWS_API_KEY', '')
    except json.JSONDecodeError as e:
        logger.error(f"api_config.json is malformed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

TELEGRAM_TOKEN, ALLOWED_CHAT_ID, NEWS_API_KEY = load_config()

# --- Constants ---
KRAKEN_OHLC_URL = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1'
HISTORY_FILE = "price_history.csv"
MAX_HISTORY = 100
MIN_ANALYSIS_DURATION = 5
DEFAULT_ANALYSIS_DURATION = 50
MAX_API_RETRIES = 5
API_RETRY_DELAY = 5

# --- Global Variables ---
price_history = []
volume_history = []
high_history = []
low_history = []
open_history = []
support_level = None
resistance_level = None
chat_locks = {}  # Dictionary to track analysis status per chat_id
analysis_durations = {}
authorized_chats = {}

# --- Save and Load Authorized Chats ---
def save_authorized_chats():
    try:
        with open(AUTHORIZED_CHATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(authorized_chats, f, indent=4)
        logger.info(f"Successfully saved authorized chats to {AUTHORIZED_CHATS_FILE}")
    except Exception as e:
        logger.error(f"Error saving {AUTHORIZED_CHATS_FILE}: {e}")

def load_authorized_chats():
    global authorized_chats
    try:
        if os.path.exists(AUTHORIZED_CHATS_FILE):
            with open(AUTHORIZED_CHATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.error(f"{AUTHORIZEd_CHATS_FILE} is not a valid JSON object")
                    authorized_chats = {}
                else:
                    authorized_chats = {str(k): v for k, v in data.items()}
            logger.info(f"Loaded authorized chats: {authorized_chats}")
        else:
            logger.info(f"{AUTHORIZEd_CHATS_FILE} does not exist")
            authorized_chats = {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding {AUTHORIZEd_CHATS_FILE}: {e}")
        authorized_chats = {}
    except Exception as e:
        logger.error(f"Error loading {AUTHORIZEd_CHATS_FILE}: {e}")
        authorized_chats = {}

# --- Check if chat_id is allowed and user is admin in group ---
async def is_allowed_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    try:
        chat_id = str(update.effective_chat.id)
        user_id = update.effective_user.id

        if chat_id == ALLOWED_CHAT_ID:
            return True

        if chat_id in authorized_chats:
            key = authorized_chats[chat_id].get("key")
            if key:
                is_valid, message = is_key_valid(key, chat_id)
                if is_valid:
                    if update.effective_chat.type in ['group', 'supergroup']:
                        try:
                            member = await update.effective_chat.get_member(user_id)
                            if member.status not in ['administrator', 'creator']:
                                await update.message.reply_text("Chỉ quản trị viên của nhóm mới có thể sử dụng các lệnh của bot.")
                                logger.warning(f"User {user_id} in chat {chat_id} is not an admin.")
                                return False
                        except TelegramError as e:
                            logger.error(f"Error checking admin status for user {user_id} in chat {chat_id}: {e}")
                            await update.message.reply_text("Lỗi khi kiểm tra quyền quản trị viên. Vui lòng liên hệ zalo: 0989916741.")
                            return False
                    return True
                else:
                    del authorized_chats[chat_id]
                    save_authorized_chats()
                    await update.message.reply_text(f"{message} Vui lòng nhập lại key bằng lệnh /key <key>. Liên hệ zalo: 0989916741.")
                    return False

        await update.message.reply_text("Đoạn chat này không có quyền sử dụng bot. Vui lòng nhập key bằng lệnh /key <key>. Liên hệ zalo: 0989916741.")
        return False
    except Exception as e:
        logger.error(f"Error in is_allowed_chat for chat {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi kiểm tra quyền truy cập. Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")
        return False

# --- Key Command ---
async def key_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = str(update.effective_chat.id)
        current_time = time.time()
        
        if chat_id == ALLOWED_CHAT_ID:
            await update.message.reply_text("Đoạn chat này đã được cấp quyền mặc định và không cần nhập key.")
            return
        
        if len(context.args) != 1:
            await update.message.reply_text("Vui lòng cung cấp key. Ví dụ: /key ABC")
            return
        
        provided_key = context.args[0].strip()
        if not provided_key:
            await update.message.reply_text("Key không được để trống.")
            return

        is_valid, message = is_key_valid(provided_key, chat_id)
        
        if chat_id in authorized_chats and authorized_chats[chat_id]["key_attempts"] >= 2:
            await update.message.reply_text("Đoạn chat này đã bị khóa do nhập key quá số lần cho phép. Liên hệ zalo: 0989916741 để được hỗ trợ.")
            logger.warning(f"Chat {chat_id} is blocked due to multiple key attempts.")
            return
        
        if is_valid:
            authorized_chats[chat_id] = {"key": provided_key, "timestamp": current_time, "key_attempts": 0}
            save_authorized_chats()
            await update.message.reply_text(f"Key hợp lệ! Đoạn chat này đã được cấp quyền sử dụng bot.")
            logger.info(f"Chat {chat_id} authorized with key: {provided_key}")
        else:
            authorized_chats[chat_id] = authorized_chats.get(chat_id, {"key": None, "timestamp": current_time, "key_attempts": 0})
            authorized_chats[chat_id]["key_attempts"] += 1
            save_authorized_chats()
            await update.message.reply_text(f"{message} Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")
            logger.warning(f"Chat {chat_id} attempted to use invalid key: {provided_key}")
    except Exception as e:
        logger.error(f"Error processing key command in chat {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi xử lý key. Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")

# --- Get Help Text ---
def get_help_text():
    return """
Danh sách các lệnh hỗ trợ:
- /start: Khởi động bot và nhận thông tin chào mừng.
- /key <key>: Nhập key để cấp quyền sử dụng bot cho đoạn chat này (ví dụ: /key ABC).
- /analysis: Phân tích thị trường BTC/USD và đưa ra tín hiệu giao dịch.
- /second <giây>: Cài đặt thời gian phân tích cho lệnh /analysis (tối thiểu 5 giây). Ví dụ: /second 50
- /cskh: Liên hệ hỗ trợ qua Zalo
- /help: Hiển thị danh sách các lệnh này

Gõ /<lệnh> để sử dụng!
Lưu ý: Trong nhóm, chỉ quản trị viên mới có thể sử dụng các lệnh của bot. Key chỉ được dùng cho 1 đoạn chat.
"""

# --- Start Command ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        welcome_message = "Chào mừng bạn đến với bot trade! Sử dụng lệnh /help để xem danh sách các lệnh hỗ trợ."
        await update.message.reply_text(welcome_message)
        logger.info(f"Start command executed in chat {update.effective_chat.id}")
    except TelegramError as e:
        logger.error(f"Error sending start message to chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("Lỗi khi gửi thông báo chào mừng. Vui lòng thử lại.")

# --- Analysis Command ---
async def analysis_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not await is_allowed_chat(update, context):
            return
        chat_id = str(update.effective_chat.id)
        
        # Check if chat is already analyzing
        if chat_id in chat_locks and chat_locks[chat_id].locked():
            await update.message.reply_text("Phân tích đang diễn ra cho đoạn chat này, vui lòng đợi.")
            return

        # Create a lock for this chat_id if it doesn't exist
        if chat_id not in chat_locks:
            chat_locks[chat_id] = asyncio.Lock()

        analysis_duration = analysis_durations.get(chat_id, DEFAULT_ANALYSIS_DURATION)
        
        # Run analysis in a separate task to allow parallel processing
        async with chat_locks[chat_id]:
            await analyze_market(update, context, analysis_duration)
    except Exception as e:
        logger.error(f"Error in analysis command for chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("Lỗi khi thực hiện phân tích. Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")

# --- Second Command ---
async def second_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not await is_allowed_chat(update, context):
            return
        chat_id = str(update.effective_chat.id)
        if len(context.args) != 1:
            await update.message.reply_text("Vui lòng cung cấp thời gian phân tích (giây). Ví dụ: /second 50")
            return
        try:
            duration = int(context.args[0].strip())
            if duration < MIN_ANALYSIS_DURATION:
                await update.message.reply_text(f"Thời gian phân tích phải lớn hơn hoặc bằng {MIN_ANALYSIS_DURATION} giây.")
                return
            analysis_durations[chat_id] = duration
            await update.message.reply_text(f"Đã cài đặt thời gian phân tích cho /analysis là {duration} giây.")
            logger.info(f"Analysis duration for chat {chat_id} set to {duration} seconds")
        except ValueError:
            await update.message.reply_text("Vui lòng nhập một số nguyên hợp lệ cho thời gian (giây).")
    except Exception as e:
        logger.error(f"Error in second command for chat {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi cài đặt thời gian phân tích. Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")

# --- CSKH Command ---
async def cskh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not await is_allowed_chat(update, context):
            return
        support_message = "Cần hỗ trợ? Liên hệ qua zalo: 0989916741"
        await update.message.reply_text(support_message)
    except Exception as e:
        logger.error(f"Error in cskh command for chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("Lỗi khi gửi thông tin hỗ trợ. Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")

# --- Help Command ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not await is_allowed_chat(update, context):
            return
        help_text = get_help_text()
        await update.message.reply_text(help_text)
    except Exception as e:
        logger.error(f"Error in help command for chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("Lỗi khi hiển thị trợ giúp. Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")

# --- Get BTC Price and Volume with aiohttp ---
async def get_btc_price_and_volume():
    global price_history, volume_history, high_history, low_history, open_history
    async with aiohttp.ClientSession() as session:
        for attempt in range(MAX_API_RETRIES):
            try:
                async with session.get(KRAKEN_OHLC_URL, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if 'error' in data and data['error']:
                        logger.warning(f"Kraken API error: {data['error']}")
                        raise ValueError("Kraken API returned error")
                    
                    if 'result' not in data or 'XXBTZUSD' not in data['result']:
                        logger.error("Invalid Kraken API response structure")
                        raise ValueError("Invalid API response structure")
                    
                    klines = data['result']['XXBTZUSD']
                    if not klines or len(klines) < MAX_HISTORY:
                        logger.warning(f"Kraken data insufficient: {len(klines)} candles received")
                        raise ValueError("Insufficient data")
                    
                    prices = [float(candle[4]) for candle in klines[:MAX_HISTORY]]
                    volumes = [float(candle[5]) for candle in klines[:MAX_HISTORY]]
                    highs = [float(candle[2]) for candle in klines[:MAX_HISTORY]]
                    lows = [float(candle[3]) for candle in klines[:MAX_HISTORY]]
                    opens = [float(candle[1]) for candle in klines[:MAX_HISTORY]]
                    latest_price = prices[-1]
                    latest_volume = volumes[-1]
                    
                    price_history = prices[-MAX_HISTORY:]
                    volume_history = volumes[-MAX_HISTORY:]
                    high_history = highs[-MAX_HISTORY:]
                    low_history = lows[-MAX_HISTORY:]
                    open_history = opens[-MAX_HISTORY:]
                    
                    return latest_price, latest_volume, prices, volumes, highs, lows, opens
            except (aiohttp.ClientError, ValueError, KeyError, IndexError) as e:
                logger.error(f"Kraken API error (attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
                if attempt < MAX_API_RETRIES - 1:
                    await asyncio.sleep(API_RETRY_DELAY)
                continue
            except Exception as e:
                logger.error(f"Unexpected error in get_btc_price_and_volume: {e}")
                break
        return None, None, None, None, None, None, None

# --- Calculate VWAP ---
def calculate_vwap(prices, volumes, highs, lows, period=14):
    try:
        if len(prices) < period or len(volumes) < period:
            return None
        typical_prices = [(highs[i] + lows[i] + prices[i]) / 3 for i in range(-period, 0)]
        vwap = sum(typical_prices[i] * volumes[i] for i in range(-period, 0)) / sum(volumes[-period:])
        return vwap
    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        return None

# --- Calculate ATR ---
def calculate_atr(highs, lows, closes, period=14):
    try:
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
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None

# --- Calculate Fibonacci Retracement Levels ---
def calculate_fibonacci_levels(prices, period=100):
    try:
        if len(prices) < period:
            return None, None, None
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
    try:
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return None

# --- Calculate EMA ---
def calculate_ema(prices, period):
    try:
        if len(prices) < period or not prices:
            return None
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
    try:
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
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return None, None

# --- Calculate RSI ---
def calculate_rsi(prices, period=14):
    try:
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
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return np.nan

# --- Calculate Stochastic RSI ---
def calculate_stochastic_rsi(prices, period=14, smooth_k=3, smooth_d=3):
    try:
        if len(prices) < period + 1:
            return None, None
        rsi_vals = [calculate_rsi(prices[:i+1], period) for i in range(period, len(prices))]
        if len(rsi_vals) < period:
            return None, None
        rsi_series = np.array(rsi_vals)
        lowest_rsi = np.min(rsi_series[-period:])
        highest_rsi = np.max(rsi_series[-period:])
        if highest_rsi == lowest_rsi:
            stoch_rsi = 0
        else:
            stoch_rsi = (rsi_series[-1] - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
        k = np.mean([stoch_rsi] + [((rsi_series[i] - np.min(rsi_series[i-period:i])) / 
                                   (np.max(rsi_series[i-period:i]) - np.min(rsi_series[i-period:i]) + 1e-10)) * 100 
                                   for i in range(-smooth_k+1, 0)]) if len(rsi_series) >= smooth_k else stoch_rsi
        d = np.mean([k] + [np.mean([((rsi_series[j] - np.min(rsi_series[j-period:j])) / 
                                     (np.max(rsi_series[j-period:j]) - np.min(rsi_series[j-period:j]) + 1e-10)) * 100 
                                     for j in range(i-period, i)]) for i in range(-smooth_d+1, 0)]) if len(rsi_series) >= smooth_d else k
        return k, d
    except Exception as e:
        logger.error(f"Error calculating Stochastic RSI: {e}")
        return None, None

# --- Calculate CCI ---
def calculate_cci(highs, lows, closes, period=20):
    try:
        if len(closes) < period:
            return None
        typical_price = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(-period, 0)]
        sma_tp = np.mean(typical_price)
        mean_dev = np.mean([abs(tp - sma_tp) for tp in typical_price])
        if mean_dev == 0:
            return 0
        cci = (typical_price[-1] - sma_tp) / (0.015 * mean_dev)
        return cci
    except Exception as e:
        logger.error(f"Error calculating CCI: {e}")
        return None

# --- Calculate Bollinger Bands ---
def calculate_bollinger_bands(prices, period=20, num_std=2):
    try:
        if len(prices) < period:
            return None, None, None
        sma = sum(prices[-period:]) / period
        std = np.std(prices[-period:])
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        return sma, upper_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return None, None, None

# --- Calculate Parabolic SAR ---
def calculate_parabolic_sar(highs, lows, closes, af=0.02, max_af=0.2):
    try:
        if len(closes) < 2:
            return None
        sar = [lows[-2]]
        ep = highs[-2]
        af_current = af
        trend = 1 if closes[-1] > closes[-2] else -1
        for i in range(-len(closes)+2, 0):
            if trend > 0:
                sar.append(sar[-1] + af_current * (ep - sar[-1]))
                if sar[-1] > lows[i]:
                    sar[-1] = min(lows[i], lows[i-1])
                    trend = -1
                    ep = lows[i]
                    af_current = af
                else:
                    if highs[i] > ep:
                        ep = highs[i]
                        af_current = min(af_current + af, max_af)
            else:
                sar.append(sar[-1] + af_current * (ep - sar[-1]))
                if sar[-1] < highs[i]:
                    sar[-1] = max(highs[i], highs[i-1])
                    trend = 1
                    ep = highs[i]
                    af_current = af
                else:
                    if lows[i] < ep:
                        ep = lows[i]
                        af_current = min(af_current + af, max_af)
        return sar[-1]
    except Exception as e:
        logger.error(f"Error calculating Parabolic SAR: {e}")
        return None

# --- Calculate ADX ---
def calculate_adx(highs, lows, closes, period=14):
    try:
        if len(closes) < period + 1:
            return None
        dm_plus = []
        dm_minus = []
        tr_list = []
        for i in range(-period, 0):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            dm_plus.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            dm_minus.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
        smoothed_tr = sum(tr_list) / period
        smoothed_plus = sum(dm_plus) / period
        smoothed_minus = sum(dm_minus) / period
        di_plus = (smoothed_plus / smoothed_tr) * 100 if smoothed_tr != 0 else 0
        di_minus = (smoothed_minus / smoothed_tr) * 100 if smoothed_tr != 0 else 0
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) != 0 else 0
        adx = dx
        return adx
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        return None

# --- Calculate OBV ---
def calculate_obv(closes, volumes):
    try:
        if len(closes) < 2:
            return None
        obv = [0]
        for i in range(-len(closes)+1, 0):
            if closes[i] > closes[i-1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        return obv[-1]
    except Exception as e:
        logger.error(f"Error calculating OBV: {e}")
        return None

# --- Calculate Volume Profile (POC, VAH, VAL) ---
def calculate_volume_profile(prices, volumes, period=20):
    try:
        if len(prices) < period or len(volumes) < period:
            return None, None, None
        price_bins = np.histogram(prices[-period:], bins=20, weights=volumes[-period:])[1]
        volume_bins = np.histogram(prices[-period:], bins=20, weights=volumes[-period:])[0]
        poc_idx = np.argmax(volume_bins)
        poc = (price_bins[poc_idx] + price_bins[poc_idx+1]) / 2
        total_volume = sum(volume_bins)
        value_area_volume = total_volume * 0.7
        sorted_volumes = sorted(zip(volume_bins, price_bins[:-1]), reverse=True)
        cumulative_volume = 0
        vah, val = None, None
        for vol, price in sorted_volumes:
            cumulative_volume += vol
            if cumulative_volume >= value_area_volume:
                vah = price_bins[list(volume_bins).index(vol) + 1]
                val = price_bins[list(volume_bins).index(vol)]
                break
        return poc, vah, val
    except Exception as e:
        logger.error(f"Error calculating volume profile: {e}")
        return None, None, None

# --- Detect Candlestick Patterns ---
def detect_candlestick_pattern(highs, lows, opens, closes):
    try:
        if len(closes) < 3:
            return None
        body = abs(opens[-1] - closes[-1])
        range_candle = highs[-1] - lows[-1]
        prev_body = abs(opens[-2] - closes[-2])
        prev_range = highs[-2] - lows[-2]
        
        if body <= range_candle * 0.1:
            return "Doji - Thị trường lưỡng lự, cảnh báo đảo chiều"
        
        if prev_body > prev_range * 0.3 and body > range_candle * 0.5:
            if closes[-2] < opens[-2] and closes[-1] > opens[-1] and closes[-1] > opens[-2] and opens[-1] < closes[-2]:
                return "Bullish Engulfing - Đảo chiều tăng mạnh"
            if closes[-2] > opens[-2] and closes[-1] < opens[-1] and closes[-1] < opens[-2] and opens[-1] > closes[-2]:
                return "Bearish Engulfing - Đảo chiều giảm mạnh"
        
        lower_wick = opens[-1] - lows[-1] if opens[-1] > closes[-1] else closes[-1] - lows[-1]
        upper_wick = highs[-1] - opens[-1] if opens[-1] > closes[-1] else highs[-1] - closes[-1]
        if body < range_candle * 0.3:
            if lower_wick > body * 2 and upper_wick < body:
                return "Hammer - Đảo chiều tăng ở đáy"
            if upper_wick > body * 2 and lower_wick < body:
                return "Shooting Star - Đảo chiều giảm ở đỉnh"
            if lower_wick > body * 2 or upper_wick > body * 2:
                return "Pin Bar - Đảo chiều tiềm năng"
        
        if len(closes) >= 3:
            if (closes[-3] < opens[-3] and abs(opens[-2] - closes[-2]) < prev_range * 0.2 and 
                closes[-1] > opens[-1] and closes[-1] > closes[-3]):
                return "Morning Star - Đảo chiều tăng sau xu hướng giảm"
            if (closes[-3] > opens[-3] and abs(opens[-2] - closes[-2]) < prev_range * 0.2 and 
                closes[-1] < opens[-1] and closes[-1] < closes[-3]):
                return "Evening Star - Đảo chiều giảm sau xu hướng tăng"
        
        if highs[-1] < highs[-2] and lows[-1] > lows[-2]:
            return "Inside Bar - Chờ breakout"
        
        if len(closes) >= 3:
            if (closes[-1] > opens[-1] and closes[-2] > opens[-2] and closes[-3] > opens[-3] and 
                closes[-1] > closes[-2] and closes[-2] > closes[-3]):
                return "Three White Soldiers - Tiếp diễn tăng"
            if (closes[-1] < opens[-1] and closes[-2] < opens[-2] and closes[-3] < opens[-3] and 
                closes[-1] < closes[-2] and closes[-2] < closes[-3]):
                return "Three Black Crows - Tiếp diễn giảm"
        
        if body > range_candle * 0.9:
            if closes[-1] > opens[-1]:
                return "Bullish Marubozu - Áp lực mua mạnh"
            else:
                return "Bearish Marubozu - Áp lực bán mạnh"
        
        return None
    except Exception as e:
        logger.error(f"Error detecting candlestick pattern: {e}")
        return None

# --- Detect Chart Patterns ---
def detect_chart_patterns(prices, highs, lows, period=50):
    try:
        if len(prices) < period:
            return None
        highs_period = highs[-period:]
        lows_period = lows[-period:]
        
        max_highs = sorted([(h, i) for i, h in enumerate(highs_period)], reverse=True)[:2]
        min_lows = sorted([(l, i) for i, l in enumerate(lows_period)])[:2]
        if len(max_highs) >= 2 and abs(max_highs[0][0] - max_highs[1][0]) < max_highs[0][0] * 0.01 and abs(max_highs[0][1] - max_highs[1][1]) > 5:
            return "Double Top - Đảo chiều giảm"
        if len(min_lows) >= 2 and abs(min_lows[0][0] - min_lows[1][0]) < min_lows[0][0] * 0.01 and abs(min_lows[0][1] - min_lows[1][1]) > 5:
            return "Double Bottom - Đảo chiều tăng"
        
        return None
    except Exception as e:
        logger.error(f"Error detecting chart patterns: {e}")
        return None

# --- Calculate Volume Spike ---
def calculate_volume_spike(volumes, period=5, threshold=1.5):
    try:
        if len(volumes) < period:
            return False
        avg_volume = np.mean(volumes[-period:])
        return volumes[-1] > avg_volume * threshold
    except Exception as e:
        logger.error(f"Error calculating volume spike: {e}")
        return False

# --- Detect Breakout ---
def detect_breakout(prices, highs, lows, volumes, period=20):
    try:
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
        elif latest_price > dynamic_resistance * 1.005 and not volume_spike:
            return "False Breakout Up - Cảnh báo đảo chiều", False
        elif latest_price < dynamic_support * 0.995 and not volume_spike:
            return "False Breakout Down - Cảnh báo đảo chiều", False
        return None, False
    except Exception as e:
        logger.error(f"Error detecting breakout: {e}")
        return None, False

# --- Detect Support and Resistance ---
def detect_support_resistance(price):
    try:
        global support_level, resistance_level
        if support_level is None or (price is not None and price < support_level * 0.98):
            support_level = price * 0.98 if price is not None else None
        if resistance_level is None or (price is not None and price > resistance_level * 1.02):
            resistance_level = price * 1.02 if price is not None else None
        if price is not None and support_level is not None and price <= support_level:
            return "Chạm vùng hỗ trợ - Cơ hội MUA"
        elif price is not None and resistance_level is not None and price >= resistance_level:
            return "Chạm vùng kháng cự - Cơ hội BÁN"
        return "Ổn định"
    except Exception as e:
        logger.error(f"Error detecting support/resistance: {e}")
        return "Ổn định"

# --- Detect Volume Spike Reversal ---
def detect_volume_spike_reversal(prices, volumes):
    try:
        if len(prices) < 5 or len(volumes) < 5:
            return None
        if calculate_volume_spike(volumes, period=5, threshold=2.0):
            if prices[-1] < prices[-2] and volumes[-1] > volumes[-2]:
                return "Volume Spike Reversal - Đảo chiều tăng"
            elif prices[-1] > prices[-2] and volumes[-1] > volumes[-2]:
                return "Volume Spike Reversal - Đảo chiều giảm"
        return None
    except Exception as e:
        logger.error(f"Error detecting volume spike reversal: {e}")
        return None

# --- Detect Wick Hunting ---
def detect_wick_hunting(highs, lows, opens, closes):
    try:
        if len(closes) < 2:
            return None
        body = abs(opens[-1] - closes[-1])
        lower_wick = min(opens[-1], closes[-1]) - lows[-1]
        upper_wick = highs[-1] - max(opens[-1], closes[-1])
        if lower_wick > body * 2:
            return "Wick Hunting - Vào lệnh BÁN"
        if upper_wick > body * 2:
            return "Wick Hunting - Vào lệnh MUA"
        return None
    except Exception as e:
        logger.error(f"Error detecting wick hunting: {e}")
        return None

# --- Detect BB Squeeze Breakout ---
def detect_bb_squeeze_breakout(prices, period=20, num_std=2):
    try:
        if len(prices) < period:
            return None
        sma, upper_band, lower_band = calculate_bollinger_bands(prices, period, num_std)
        if upper_band is None or lower_band is None:
            return None
        bandwidth = (upper_band - lower_band) / sma
        if bandwidth < 0.02 and prices[-1] > upper_band:
            return "BB Squeeze Breakout - Tăng mạnh"
        elif bandwidth < 0.02 and prices[-1] < lower_band:
            return "BB Squeeze Breakout - Giảm mạnh"
        return None
    except Exception as e:
        logger.error(f"Error detecting BB squeeze breakout: {e}")
        return None

# --- Detect Fractal ---
def detect_fractal(highs, highs, window=5):
    try:
        if len(highs) < window:
            return None
        mid_idx = window // 2
        if (highs[-mid_idx] == max(highs[-window:]) and highs[-mid_idx] > highs[-mid_idx-1] and highs[-mid_idx] > highs[-mid_idx+1]):
            return "Fractal High - Kháng cự tiềm năng"
        if (lows[-mid_idx] == min(lows[-window:]) and lows[-mid_idx] < lows[-mid_idx-1] and lows[-mid_idx] < lows[-mid_idx+1]):
            return "Fractal Low - Hỗ trợ tiềm năng"
        return None
    except Exception as e:
        logger.error(f"Error detecting fractal: {e}")
        return None

# --- Detect OBV Divergence ---
def detect_obv_divergence(prices, volumes):
    try:
        if len(prices) < 5 or len(volumes) < 5:
            return None
        obv = calculate_obv(prices, volumes)
        if obv is None:
            return None
        if prices[-1] < prices[-2] and obv > calculate_obv(prices[:-1], volumes[:-1]):
            return "OBV Divergence - Đảo chiều tăng"
        if prices[-1] > prices[-2] and obv < calculate_obv(prices[:-1], volumes[:-1]):
            return "OBV Divergence - Đảo chiều giảm"
        return None
    except Exception as e:
        logger.error(f"Error detecting OBV divergence: {e}")
        return None

# --- Detect RSI Divergence ---
def detect_rsi_divergence(prices, rsi_values, period=14):
    try:
        if len(prices) < period + 1 or len(rsi_values) < period + 1:
            return None
        rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]
        if prices[-1] < prices[-2] and rsi > prev_rsi:
            return "Regular RSI Divergence - Đảo chiều tăng"
        if prices[-1] > prices[-2] and rsi < prev_rsi]:
            return "Regular RSI Divergence - Đảo chiều giảm"
        if prices[-1] > prices[-2] and rsi > prev_rsi and rsi < 70:
            return "Hidden RSI Divergence - Tiếp tục tăng"
        if prices[-1] < prices[-2] and rsi < prev_rsi and rsi > 30:
            return "Hidden RSI Divergence - Tiếp tục giảm"
        return None
    except Exception as e:
        logger.error(f"Error detecting RSI divergence: {e}")
        return None

# --- Detect Order Block (OBV) ---
def detect_order_block(highs, lows, closes, volumes, period=20):
    try:
        if len(closes) < period:
            return None
        max_volume_idx = np.argmax(volumes[-period:])
        ob_price = closes[-period + max_volume_idx]
        if closes[-1] <= ob_price * 1.005 and closes[-1] >= ob_price * 0.995:
            return "Order Block - Giá quay lại vùng tổ chức"
        return None
    except Exception as e:
        logger.error(f"Error detecting order block: {e}")
        return None

# --- Detect Fair Value Gap (FVG) ---
def detect_fair_value_gap(highs, lows, opens, closes):
    try:
        if len(closes) < 3:
            return None
        if closes[-3] < opens[-3] and highs[-2] < lows[-3] and closes[-1] > opens[-1]:
            return "Bullish FVG - Giá quay lại vùng hút giá tăng"
        if closes[-3] > opens[-3] and lows[-2] > highs[-3] and closes[-1] < opens[-1]:
            return "Bearish FVG - Giá quay lại vùng hút giá giảm"
        return None
    except Exception as e:
        logger.error(f"Error detecting fair value gap: {e}")
        return None

# --- Detect Liquidity Hunt ---
def detect_liquidity_hunt(highs, lows, closes, volumes):
    try:
        if len(closes) < 3 or len(volumes) < 3:
            return None
        if volumes[-1] > np.mean(volumes[-5:-1]) * 2 and highs[-1] > max(highs[-5:-1]):
            return "Liquidity Hunt Up - Vào lệnh BÁN"
        if volumes[-1] > np.mean(volumes[-5:-1]) * 2 and lows[-1] < min(lows[-5:-1]):
            return "Liquidity Hunt Down - Vào lệnh MUA"
        return None
    except Exception as e:
        logger.error(f"Error detecting liquidity hunt: {e}")
        return None

# --- Detect Heiken Ashi Trend ---
def detect_heiken_ashi_trend(highs, lows, opens, closes):
    try:
        if len(closes) < 3:
            return None
        ha_close = [(opens[i] + highs[i] + lows[i] + closes[i]) / 4 for i in range(-3, 0)]
        ha_open = [(opens[i-1] + closes[i-1]) / 2 for i in range(-2, 0)]
        if ha_close[-1] > ha_open[-1] and ha_close[-2] > ha_open[-2]:
            return "Heiken Ashi Bullish - Xu hướng tăng"
        if ha_close[-1] < ha_open[-1] and ha_close[-2] < ha_open[-2]:
            return "Heiken Ashi Bearish - Xu hướng giảm"
        return None
    except Exception as e:
        logger.error(f"Error detecting Heiken Ashi trend: {e}")
        return None

# --- Detect Three Candle Scalping ---
def detect_three_candle_scalping(highs, lows, opens, closes, volumes):
    try:
        if len(closes) < 4:
            return None
        if (closes[-3] > opens[-3] and closes[-2] > opens[-2] and closes[-1] > opens[-1] and
            closes[-1] > closes[-2] and closes[-2] > closes[-3] and
            sum(volumes[-3:]) > sum(volumes[-6:-3])):
            return "Three Candle Bullish - Vào lệnh MUA"
        if (closes[-3] < opens[-3] and closes[-2] < opens[-2] and closes[-1] < opens[-1] and
            closes[-1] < closes[-2] and closes[-2] < closes[-3] and
            sum(volumes[-3:]) > sum(volumes[-6:-3])):
            return "Three Candle Bearish - Vào lệnh BÁN"
        return None
    except Exception as e:
        logger.error(f"Error detecting three candle scalping: {e}")
        return None

# --- Detect Round Number Levels ---
def detect_round_number_levels(price):
    try:
        round_numbers = [round(price / 1000) * 1000, round(price / 500) * 500]
        for rn in round_numbers:
            if abs(price - rn) / price < 0.005:
                return f"Round Number {rn} - Cảnh báo đảo chiều"
        return None
    except Exception as e:
        logger.error(f"Error detecting round number levels: {e}")
        return None

# --- Detect Elliott Wave ---
def detect_elliott_wave(prices, period=20):
    try:
        if len(prices) < period:
            return None
        highs = np.array(prices[-period:])
        lows = np.array(prices[-period:])
        peaks = []
        troughs = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        if len(peaks) >= 2 and len(troughs) >= 2:
            if peaks[-1][1] > peaks[-2][1] and troughs[-1][1] > troughs[-2][1]:
                return "Elliott Wave 3/5 Bullish - Vào lệnh MUA"
            if peaks[-1][1] < peaks[-2][1] and troughs[-1][1] < troughs[-2][1]:
                return "Elliott Wave 3/5 Bearish - Vào lệnh BÁN"
        return None
    except Exception as e:
        logger.error(f"Error detecting Elliott Wave: {e}")
        return None

# --- Detect Wyckoff Accumulation/Distribution ---
def detect_wyckoff_pattern(prices, volumes, period=20):
    try:
        if len(prices) < period or len(volumes) < period:
            return None
        avg_volume = np.mean(volumes[-period:])
        range_price = max(prices[-period:]) - min(prices[-period:])
        if range_price < np.std(prices[-period:]) * 1.5 and volumes[-1] < avg_volume * 0.7:
            return "Wyckoff Accumulation - Vùng tích lũy"
        if range_price < np.std(prices[-period:]) * 1.5 and volumes[-1] > avg_volume * 1.5:
            return "Wyckoff Distribution - Vùng phân phối"
        return None
    except Exception as e:
        logger.error(f"Error detecting Wyckoff pattern: {e}")
        return None

# --- Format Value ---
def format_value(value, decimals=2):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.{decimals}f}"
    except Exception as e:
        logger.error(f"Error formatting value: {e}")
        return "N/A"

# --- Save Data to CSV ---
def save_to_csv(price, trend, win_rate, market_status, chat_id):
    try:
        file_exists = os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Time', 'ChatID', 'Price', 'Trend', 'WinRate', 'MarketStatus'])
            writer.writerow([datetime.now().strftime("%H:%M:%S %d-%m-%Y"), chat_id, price, trend, win_rate, market_status])
        logger.info(f"Saved analysis data to {HISTORY_FILE} for chat {chat_id}")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

# --- Analyze Market ---
async def analyze_market(update: Update, context: ContextTypes.DEFAULT_TYPE, analysis_duration=50):
    global price_history, volume_history, high_history, low_history, open_history
    chat_id = str(update.effective_chat.id)

    start_time = time.time()
    temp_prices = []
    temp_volumes = []
    temp_highs = []
    temp_lows = []
    temp_opens = []

    try:
        while time.time() - start_time < max(analysis_duration, MIN_ANALYSIS_DURATION):
            latest_price, latest_volume, prices, volumes, highs, lows, opens = await get_btc_price_and_volume()
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
    except Exception as e:
        logger.error(f"Error in analyze_market loop for chat {chat_id}: {e}")

    if not temp_prices:
        await update.message.reply_text("Không thể lấy dữ liệu từ Coincex. Vui lòng thử lại hoặc kiểm tra kết nối mạng.")
        return

    try:
        latest_price = temp_prices[-1]
        price_history = price_history[-MAX_HISTORY:]

        # Technical analysis
        ema_9 = calculate_ema(price_history, 9)
        ema_21 = calculate_ema(price_history, 21)
        ema_20 = calculate_ema(price_history, 20)
        ema_50 = calculate_ema(price_history, 50)
        sma_20, upper_band, lower_band = calculate_bollinger_bands(price_history)
        macd, signal = calculate_macd(price_history)
        rsi = calculate_rsi(price_history)
        rsi_values = [calculate_rsi(price_history[:i+1], 14) for i in range(len(price_history))]
        stoch_k, stoch_d = calculate_stochastic_rsi(price_history)
        cci = calculate_cci(high_history, low_history, price_history)
        psar = calculate_parabolic_sar(high_history, low_history, price_history)
        adx = calculate_adx(high_history, low_history, price_history)
        obv = calculate_obv(price_history, volume_history)
        fib_382, fib_618, _ = calculate_fibonacci_levels(price_history)
        volume_spike = calculate_volume_spike(volume_history)
        poc, vah, val = calculate_volume_profile(price_history, volume_history)
        candlestick_pattern = detect_candlestick_pattern(high_history, low_history, open_history, price_history)
        chart_pattern = detect_chart_patterns(price_history, high_history, low_history)
        breakout, is_true_breakout = detect_breakout(price_history, high_history, low_history, volume_history)
        support_resistance_signal = detect_support_resistance(latest_price)
        volume_spike_reversal = detect_volume_spike_reversal(price_history, volume_history)
        wick_hunting = detect_wick_hunting(high_history, low_history, open_history, price_history)
        bb_squeeze_breakout = detect_bb_squeeze_breakout(price_history)
        fractal = detect_fractal(high_history, low_history)
        obv_divergence = detect_obv_divergence(price_history, volume_history)
        rsi_divergence = detect_rsi_divergence(price_history, rsi_values)
        order_block = detect_order_block(high_history, low_history, price_history, volume_history)
        fair_value_gap = detect_fair_value_gap(high_history, low_history, open_history, price_history)
        liquidity_hunt = detect_liquidity_hunt(high_history, low_history, price_history, volume_history)
        heiken_ashi = detect_heiken_ashi_trend(high_history, low_history, open_history, price_history)
        three_candle = detect_three_candle_scalping(high_history, low_history, open_history, price_history, volume_history)
        round_number = detect_round_number_levels(latest_price)
        elliott_wave = detect_elliott_wave(price_history)
        wyckoff_pattern = detect_wyckoff_pattern(price_history, volume_history)
        atr = calculate_atr(high_history, low_history, price_history)
        volume_trend = "TĂNG" if sum(volume_history[-5:]) > sum(volume_history[-10:-5]) else "GIẢM"

        # Market Volatility Classification
        market_type = "Ổn định"
        if atr is not None and atr > np.mean([calculate_atr(high_history[:i+1], low_history[:i+1], price_history[:i+1]) or 0 for i in range(len(price_history)-1)]) * 1.5:
            market_type = "Biến động mạnh"
        elif atr is not None and atr < np.mean([calculate_atr(high_history[:i+1], low_history[:i+1], price_history[:i+1]) or 0 for i in range(len(price_history)-1)]) * 0.7:
            market_type = "Biến động nhẹ"
        bandwidth = (upper_band - lower_band) / sma_20 if upper_band is not None and lower_band is not None and sma_20 is not None else float('inf')
        if bandwidth < 0.02:
            market_type = "Biến động nhẹ"

        # Combined strategies
        buy_signals = [
            (2.0 if rsi is not None and not np.isnan(rsi) and rsi < 30 else 0, "RSI Quá bán - Cơ hội MUA"),
            (1.5 if macd is not None and signal is not None and macd > signal else 0, "MACD Cắt lên - MUA"),
            (1.5 if ema_9 is not None and ema_21 is not None and ema_9 > ema_21 else 0, "EMA9 > EMA21 - Đảo chiều tăng"),
            (1.5 if ema_20 is not None and ema_50 is not None and ema_20 > ema_50 else 0, "EMA20 > EMA50 - Đảo chiều tăng"),
            (1.5 if lower_band is not None and latest_price <= lower_band else 0, "Chạm dải Bollinger dưới - MUA"),
            (2.0 if volume_spike else 0, "Khối lượng tăng đột biến - Xác nhận MUA"),
            (2.5 if breakout == "Breakout Up - Tín hiệu tăng mạnh" and is_true_breakout else 0, "Breakout lên - MUA mạnh"),
            (1.5 if candlestick_pattern in [
                "Bullish Engulfing - Đảo chiều tăng mạnh",
                "Hammer - Đảo chiều tăng ở đáy",
                "Morning Star - Đảo chiều tăng sau xu hướng giảm",
                "Pin Bar - Đảo chiều tiềm năng",
                "Three White Soldiers - Tiếp diễn tăng",
                "Bullish Marubozu - Áp lực mua mạnh",
            ] else 0, candlestick_pattern),
            (1.5 if chart_pattern == "Double Bottom - Đảo chiều tăng" else 0, chart_pattern),
            (1.5 if support_resistance_signal == "Chạm vùng hỗ trợ - Cơ hội MUA" else 0, support_resistance_signal),
            (2.0 if stoch_k is not None and stoch_d is not None and stoch_k < 20 and stoch_k > stoch_d else 0, "Stochastic RSI Quá bán - MUA"),
            (2.0 if cci is not None and cci < -100 else 0, "CCI Quá bán - MUA"),
            (1.5 if psar is not None and latest_price > psar else 0, "Giá trên PSAR - MUA"),
            (1.5 if adx is not None and adx > 40 and ema_20 > ema_50 else 0, "ADX mạnh, xu hướng tăng - MUA"),
            (1.5 if fib_618 is not None and abs(latest_price - fib_618) / latest_price < 0.005 else 0, "Chạm Fibonacci 0.618 - MUA"),
            (2.0 if volume_spike_reversal == "Volume Spike Reversal - Đảo chiều tăng" else 0, volume_spike_reversal),
            (2.0 if wick_hunting == "Wick Hunting - Vào lệnh MUA" else 0, wick_hunting),
            (2.5 if bb_squeeze_breakout == "BB Squeeze Breakout - Tăng mạnh" else 0, bb_squeeze_breakout),
            (1.5 if fractal == "Fractal Low - Hỗ trợ tiềm năng" else 0, fractal),
            (2.0 if obv_divergence == "OBV Divergence - Đảo chiều tăng" else 0, obv_divergence),
            (2.0 if rsi_divergence in ["Regular RSI Divergence - Đảo chiều tăng", "Hidden RSI Divergence - Tiếp diễn tăng"] else 0, rsi_divergence),
            (2.5 if order_block == "Order Block - Giá quay lại vùng tổ chức" and candlestick_pattern in ["Bullish Engulfing - Đảo chiều tăng mạnh", "Pin Bar - Đảo chiều tiềm năng"] else 0, order_block),
            (2.5 if fair_value_gap == "Bullish FVG - Giá quay lại vùng hút giá tăng" else 0, fair_value_gap),
            (2.5 if liquidity_hunt == "Liquidity Hunt Down - Vào lệnh MUA" else 0, liquidity_hunt),
            (1.5 if heiken_ashi == "Heiken Ashi Bullish - Xu hướng tăng" else 0, heiken_ashi),
            (2.0 if three_candle == "Three Candle Bullish - Vào lệnh MUA" else 0, three_candle),
            (1.5 if round_number and "Round Number" in round_number and candlestick_pattern in ["Pin Bar - Đảo chiều tiềm năng", "Doji - Thị trường lưỡng lự, cảnh báo đảo chiều"] else 0, round_number),
            (2.0 if elliott_wave == "Elliott Wave 3/5 Bullish - Vào lệnh MUA" else 0, elliott_wave),
            (1.5 if wyckoff_pattern == "Wyckoff Accumulation - Vùng tích lũy" else 0, wyckoff_pattern),
            (2.5 if market_type == "Biến động mạnh" and order_block and fair_value_gap else 0, "Liquidity Hunt + OB + FVG - MUA"),
            (2.0 if market_type == "Biến động nhẹ" and candlestick_pattern == "Inside Bar - Chờ breakout" and volume_spike else 0, "Inside Bar Breakout - MUA"),
            (2.0 if market_type == "Ổn định" and ema_9 > ema_21 and rsi < 70 and candlestick_pattern in ["Bullish Engulfing - Đảo chiều tăng mạnh", "Pin Bar - Đảo chiều tiềm năng"] else 0, "EMA + RSI + Engulfing - MUA"),
            (2.0 if poc is not None and abs(latest_price - poc) / latest_price < 0.005 and candlestick_pattern in ["Bullish Engulfing - Đảo chiều tăng mạnh", "Pin Bar - Đảo chiều tiềm năng"] else 0, "Volume Profile POC - MUA"),
        ]
        
        sell_signals = [
            (2.0 if rsi is not None and not np.isnan(rsi) and rsi > 70 else 0, "RSI Quá mua - Cơ hội BÁN"),
            (1.5 if macd is not None and signal is not None and macd < signal else 0, "MACD Cắt xuống - BÁN"),
            (1.5 if ema_9 is not None and ema_21 is not None and ema_9 < ema_21 else 0, "EMA9 < EMA21 - Đảo chiều giảm"),
            (1.5 if ema_20 is not None and ema_50 is not None and ema_20 < ema_50 else 0, "EMA20 < EMA50 - Đảo chiều giảm"),
            (1.5 if upper_band is not None and latest_price >= upper_band else 0, "Chạm dải Bollinger trên - BÁN"),
            (2.0 if volume_spike else 0, "Khối lượng tăng đột biến - Xác nhận BÁN"),
            (2.5 if breakout == "Breakout Down - Tín hiệu giảm mạnh" and is_true_breakout else 0, "Breakout xuống - BÁN"),
            (1.5 if candlestick_pattern in [
                "Bearish Engulfing - Đảo chiều giảm mạnh",
                "Shooting Star - Đảo chiều giảm ở đỉnh",
                "Evening Star - Đảo chiều giảm sau xu hướng tăng",
                "Pin Bar - Đảo chiều tiềm năng",
                "Three Black Crows - Tiếp diễn giảm",
                "Bearish Marubozu - Áp lực bán mạnh",
            ] else 0, candlestick_pattern),
            (1.5 if chart_pattern == "Double Top - Đảo chiều giảm" else 0, chart_pattern),
            (1.5 if support_resistance_signal == "Chạm vùng kháng cự - Cơ hội BÁN" else 0, support_resistance_signal),
            (2.0 if stoch_k is not None and stoch_d is not None and stoch_k > 80 and stoch_k < stoch_d else 0, "Stochastic RSI Quá mua - BÁN"),
            (2.0 if cci is not None and cci > 100 else 0, "CCI Quá mua - BÁN"),
            (1.5 if psar is not None and latest_price < psar else 0, "Giá dưới PSAR - BÁN"),
            (1.5 if adx is not None and adx > 40 and ema_20 < ema_50 else 0, "ADX mạnh, xu hướng giảm - BÁN"),
            (1.5 if fib_382 is not None and abs(latest_price - fib_382) / latest_price < 0.005 else 0, "Chạm Fibonacci 0.382 - BÁN"),
            (2.0 if volume_spike_reversal == "Volume Spike Reversal - Đảo chiều giảm" else 0, volume_spike_reversal),
            (2.0 if wick_hunting == "Wick Hunting - Vào lệnh BÁN" else 0, wick_hunting),
            (2.5 if bb_squeeze_breakout == "BB Squeeze Breakout - Giảm mạnh" else 0, bb_squeeze_breakout),
            (1.5 if fractal == "Fractal High - Kháng cự tiềm năng" else 0, fractal),
            (2.0 if obv_divergence == "OBV Divergence - Đảo chiều giảm" else 0, obv_divergence),
            (2.0 if rsi_divergence in ["Regular RSI Divergence - Đảo chiều giảm", "Hidden RSI Divergence - Tiếp diễn giảm"] else 0, rsi_divergence),
            (2.5 if order_block == "Order Block - Giá quay lại vùng tổ chức" and candlestick_pattern in ["Bearish Engulfing - Đảo chiều giảm mạnh", "Pin Bar - Đảo chiều tiềm năng"] else 0, order_block),
            (2.5 if fair_value_gap == "Bearish FVG - Giá quay lại vùng hút giá giảm" else 0, fair_value_gap),
            (2.5 if liquidity_hunt == "Liquidity Hunt Up - Vào lệnh BÁN" else 0, liquidity_hunt),
            (1.5 if heiken_ashi == "Heiken Ashi Bearish - Xu hướng giảm" else 0, heiken_ashi),
            (2.0 if three_candle == "Three Candle Bearish - Vào lệnh BÁN" else 0, three_candle),
            (1.5 if round_number and "Round Number" in round_number and candlestick_pattern in ["Pin Bar - Đảo chiều tiềm năng", "Doji - Thị trường lưỡng lự, cảnh báo đảo chiều"] else 0, round_number),
            (2.0 if elliott_wave == "Elliott Wave 3/5 Bearish - Vào lệnh BÁN" else 0, elliott_wave),
            (1.5 if wyckoff_pattern == "Wyckoff Distribution - Vùng phân phối" else 0, wyckoff_pattern),
            (2.5 if market_type == "Biến động mạnh" and order_block and fair_value_gap else 0, "Liquidity Hunt + OB + FVG - BÁN"),
            (2.0 if market_type == "Biến động nhẹ" and candlestick_pattern == "Inside Bar - Chờ breakout" and volume_spike else 0, "Inside Bar Breakout - BÁN"),
            (2.0 if market_type == "Ổn định" and ema_9 < ema_21 and rsi > 30 and candlestick_pattern in ["Bearish Engulfing - Đảo chiều giảm mạnh", "Pin Bar - Đảo chiều tiềm năng"] else 0, "EMA + RSI + Engulfing - BÁN"),
            (2.0 if poc is not None and abs(latest_price - poc) / latest_price < 0.005 and candlestick_pattern in ["Bearish Engulfing - Đảo chiều giảm mạnh", "Pin Bar - Đảo chiều tiềm năng"] else 0, "Volume Profile POC - BÁN"),
        ]

        buy_score = sum(weight for weight, _ in buy_signals)
        sell_score = sum(weight for weight, _ in sell_signals)

        trend = "MUA" if buy_score > sell_score else "BÁN" if sell_score > buy_score else "CHỜ LỆNH"
        total_score = buy_score + sell_score
        win_rate = ((buy_score / total_score) * 100 if trend == "MUA" else (sell_score / total_score) * 100 if trend == "BÁN" else 50) if total_score > 0 else 50

        latest_price_str = format_value(latest_price)
        win_rate_str = format_value(win_rate)
        market_status = f"Loại thị trường: {market_type}\n"
        market_status += f"{support_resistance_signal}\n"
        if candlestick_pattern:
            market_status += f"Nến: {candlestick_pattern}\n"
        if chart_pattern:
            market_status += f"Mô hình: {chart_pattern}\n"
        if breakout:
            market_status += f"Breakout: {breakout}\n"
        if volume_spike_reversal:
            market_status += f"Volume Spike: {volume_spike_reversal}\n"
        if wick_hunting:
            market_status += f"Wick Hunting: {wick_hunting}\n"
        if bb_squeeze_breakout:
            market_status += f"BB Squeeze: {bb_squeeze_breakout}\n"
        if fractal:
            market_status += f"Fractal: {fractal}\n"
        if obv_divergence:
            market_status += f"OBV Divergence: {obv_divergence}\n"
        if rsi_divergence:
            market_status += f"RSI Divergence: {rsi_divergence}\n"
        if order_block:
            market_status += f"Order Block: {order_block}\n"
        if fair_value_gap:
            market_status += f"Fair Value Gap: {fair_value_gap}\n"
        if liquidity_hunt:
            market_status += f"Liquidity Hunt: {liquidity_hunt}\n"
        if heiken_ashi:
            market_status += f"Heiken Ashi: {heiken_ashi}\n"
        if three_candle:
            market_status += f"Three Candle: {three_candle}\n"
        if round_number:
            market_status += f"Round Number: {round_number}\n"
        if elliott_wave:
            market_status += f"Elliott Wave: {elliott_wave}\n"
        if wyckoff_pattern:
            market_status += f"Wyckoff Pattern: {wyckoff_pattern}\n"
        if poc:
            market_status += f"POC: {format_value(poc)}\n"
        market_status += f"Khối lượng: {volume_trend}"

        report = f"""
📈 **COINCEX - BTC/USD** 📈
🕒 **Thời gian**: {datetime.now().strftime('%H:%M:%S %d-%m-%Y')}
💹 **Giá hiện tại**: {latest_price_str} USD
🚨 **Tín hiệu hệ thống**: {trend}
🎯 **Tỷ lệ thắng lợi**: {win_rate_str}%
📋 **Trạng thái**: {market_status}
"""

        await update.message.reply_text(report)
        logger.info(f"Analysis report sent to chat {chat_id}")
        save_to_csv(latest_price, trend, win_rate, market_status, chat_id)
    except TelegramError as e:
        logger.error(f"Error sending Telegram message to {chat_id}: {e}")
        await update.message.reply_text("Lỗi hệ thống khi gửi báo cáo. Vui lòng thử lại.")
    except Exception as e:
        logger.error(f"Error in analyze_market for chat {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi phân tích dữ liệu thị trường. Vui lòng thử lại hoặc liên hệ zalo: 0989916741.")

# --- Start Bot ---
async def start_bot(app: Application):
    try:
        logger.info(f"Bot started successfully. Allowed chat_id: {ALLOWED_CHAT_ID}")
        await app.initialize()
        await app.updater.start_polling(poll_interval=1.0, timeout=10, drop_pending_updates=True)
        await app.start()
        await asyncio.Event().wait()
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise
    finally:
        await app.stop()

# --- Main Function ---
async def main():
    try:
        print("Loading authorized chats...")
        load_authorized_chats()
        
        print("Initializing Telegram bot...")
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        logger.info(f"Bot initialized with TELEGRAM_TOKEN")

        print("Bot initialized. Adding handlers...")
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("analysis", analysis_command))
        application.add_handler(CommandHandler("second", second_command))
        application.add_handler(CommandHandler("cskh", cskh_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("key", key_command))
        print("Handlers added.")

        await start_bot(application)
    except TelegramError as e:
        logger.error(f"Telegram bot error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
