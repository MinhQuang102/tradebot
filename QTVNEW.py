import asyncio
import requests
import json
import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from telegram.error import TelegramError
import time
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

# --- Thiết lập Logging ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Khởi tạo Lock cho phân tích đồng thời ---
analysis_lock = asyncio.Lock()

# --- Thiết lập biến môi trường trực tiếp trong mã ---
TELEGRAM_TOKEN = "7608384401:AAHKfX5KlBl5CZTaoKSDwwdATmbY8Z34vRk"  # Thay bằng token thực tế
ALLOWED_CHAT_ID = "-1002554202438"  # Thay bằng chat ID thực tế
VALID_KEY = "10092006"  # Key xác thực

# Kiểm tra biến môi trường
if not TELEGRAM_TOKEN or not ALLOWED_CHAT_ID:
    logger.error("TELEGRAM_TOKEN hoặc ALLOWED_CHAT_ID không được thiết lập")
    raise ValueError("Thiếu TELEGRAM_TOKEN hoặc ALLOWED_CHAT_ID")

# --- Hằng số ---
KRAKEN_OHLC_URL = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1'
AUTHORIZED_CHATS_FILE = "authorized_chats.json"
HISTORY_FILE = "price_history.csv"
MAX_HISTORY = 100

# --- Biến toàn cục ---
price_history = []
volume_history = []
high_history = []
low_history = []
open_history = []
support_level = None
resistance_level = None
authorized_chats = {}

# --- Lưu và tải danh sách chat được ủy quyền ---
def save_authorized_chats():
    try:
        with open(AUTHORIZED_CHATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(authorized_chats, f)
    except Exception as e:
        logger.error(f"Lỗi khi lưu authorized_chats: {e}")

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
    except json.JSONDecodeError:
        logger.error(f"File {AUTHORIZED_CHATS_FILE} bị hỏng, khởi tạo lại.")
        authorized_chats = {}
        save_authorized_chats()
    except Exception as e:
        logger.error(f"Lỗi khi tải authorized_chats: {e}")
        authorized_chats = {}

# --- Thông báo lỗi cho admin ---
async def notify_error(application: Application, chat_id: str, error: str):
    if not application:
        logger.warning("Application là None, không thể gửi thông báo lỗi")
        return
    try:
        await application.bot.send_message(chat_id=chat_id, text=f"Lỗi bot: {error}")
    except TelegramError as e:
        logger.error(f"Không thể gửi thông báo lỗi: {e}")

# --- Kiểm tra chat_id có được phép và user là admin trong nhóm ---
async def is_allowed_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_id = str(update.effective_chat.id)
    user_id = update.effective_user.id
    current_time = time.time()
    
    if chat_id == ALLOWED_CHAT_ID:
        return True
    
    if authorized_chats and chat_id in authorized_chats:
        auth_info = authorized_chats[chat_id]
        if auth_info.get("banned", False):
            await update.message.reply_text("Đoạn chat này đã bị cấm do nhập sai key quá số lần cho phép.")
            logger.warning(f"Chat {chat_id} bị cấm.")
            return False
        if current_time - auth_info["timestamp"] < 24 * 3600:
            if update.effective_chat.type in ['group', 'supergroup']:
                try:
                    member = await update.effective_chat.get_member(user_id)
                    if member.status not in ['administrator', 'creator']:
                        await update.message.reply_text("Chỉ quản trị viên của nhóm mới có thể sử dụng các lệnh của bot.")
                        logger.warning(f"User {user_id} trong chat {chat_id} không phải admin.")
                        return False
                except TelegramError as e:
                    logger.error(f"Lỗi khi kiểm tra trạng thái admin: {e}")
                    await update.message.reply_text("Lỗi khi kiểm tra quyền quản trị viên.")
                    await notify_error(context.application, ALLOWED_CHAT_ID, f"Lỗi kiểm tra trạng thái admin trong chat {chat_id}: {e}")
                    return False
            return True
        else:
            del authorized_chats[chat_id]
            save_authorized_chats()
    
    await update.message.reply_text("Đoạn chat này không có quyền sử dụng bot. Vui lòng nhập key bằng lệnh /key <key>.")
    return False

# --- Lệnh Key ---
async def key_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    current_time = time.time()
    
    if chat_id == ALLOWED_CHAT_ID:
        await update.message.reply_text("Đoạn chat này đã được cấp quyền mặc định.")
        return
    
    if len(context.args) != 1:
        await update.message.reply_text("Vui lòng cung cấp key. Ví dụ: /key 123")
        return
    
    provided_key = context.args[0]
    try:
        if provided_key == VALID_KEY:
            authorized_chats.clear()
            authorized_chats[chat_id] = {
                "timestamp": current_time,
                "key_attempts": 1,
                "key_used": True,
                "banned": False
            }
            save_authorized_chats()
            await update.message.reply_text("Key hợp lệ! Đoạn chat này được cấp quyền trong 24 giờ.")
            logger.info(f"Chat {chat_id} được ủy quyền với key: {provided_key}")
            
            context.job_queue.run_once(
                callback=lambda ctx: remove_authorization(ctx, chat_id),
                when=24 * 3600,
                name=f"remove_auth_{chat_id}"
            )
        else:
            if chat_id not in authorized_chats:
                authorized_chats[chat_id] = {
                    "timestamp": current_time,
                    "key_attempts": 1,
                    "key_used": False,
                    "banned": False
                }
            else:
                authorized_chats[chat_id]["key_attempts"] += 1
                if authorized_chats[chat_id]["key_attempts"] >= 2:
                    authorized_chats[chat_id]["banned"] = True
            save_authorized_chats()
            await update.message.reply_text(f"Key không hợp lệ. Còn {2 - authorized_chats[chat_id]['key_attempts']} lần thử.")
            logger.warning(f"Chat {chat_id} sử dụng key không hợp lệ: {provided_key}")
    except Exception as e:
        logger.error(f"Lỗi khi xử lý lệnh key: {e}")
        await update.message.reply_text("Lỗi khi xử lý key. Vui lòng thử lại.")
        await notify_error(context.application, ALLOWED_CHAT_ID, f"Lỗi trong lệnh key cho chat {chat_id}: {e}")

async def remove_authorization(context: ContextTypes.DEFAULT_TYPE, chat_id: str):
    if chat_id in authorized_chats:
        del authorized_chats[chat_id]
        save_authorized_chats()
        try:
            await context.bot.send_message(chat_id=chat_id, text="Quyền truy cập đã hết hạn. Vui lòng nhập lại key bằng lệnh /key <key>.")
            logger.info(f"Xóa quyền truy cập cho chat {chat_id}.")
        except TelegramError as e:
            logger.error(f"Lỗi khi gửi tin nhắn hết hạn đến chat {chat_id}: {e}")
            await notify_error(context.application, ALLOWED_CHAT_ID, f"Lỗi khi gửi tin nhắn hết hạn đến chat {chat_id}: {e}")

# --- Lấy giá và khối lượng BTC ---
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def get_btc_price_and_volume():
    global price_history, volume_history, high_history, low_history, open_history
    try:
        response = requests.get(KRAKEN_OHLC_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            logger.warning(f"Lỗi API Kraken: {data['error']}")
            return None, None, None, None, None, None, None
        
        result = data.get('result', {})
        klines = result.get('XXBTZUSD', [])
        if not klines or len(klines) < 1:
            logger.warning(f"Dữ liệu Kraken rỗng: nhận được {len(klines)} nến")
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
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu Kraken: {e}")
        return None, None, None, None, None, None, None

# --- Tính VWAP ---
def calculate_vwap(prices, volumes, highs, lows, period=14):
    if len(prices) < period or len(volumes) < period:
        return None
    try:
        typical_prices = [(highs[i] + lows[i] + prices[i]) / 3 for i in range(-period, 0)]
        vwap = sum(typical_prices[i] * volumes[i] for i in range(-period, 0)) / sum(volumes[-period:])
        return vwap
    except Exception as e:
        logger.error(f"Lỗi khi tính VWAP: {e}")
        return None

# --- Tính ATR ---
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
        logger.error(f"Lỗi khi tính ATR: {e}")
        return None

# --- Tính mức Fibonacci Retracement ---
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
        logger.error(f"Lỗi khi tính Fibonacci: {e}")
        return None, None, None

# --- Tính SMA ---
def calculate_sma(prices, period=5):
    if len(prices) < period:
        return None
    try:
        return sum(prices[-period:]) / period
    except Exception as e:
        logger.error(f"Lỗi khi tính SMA: {e}")
        return None

# --- Tính EMA ---
def calculate_ema(prices, period):
    if len(prices) < period:
        return None
    try:
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    except Exception as e:
        logger.error(f"Lỗi khi tính EMA: {e}")
        return None

# --- Tính MACD ---
def calculate_macd(prices):
    if len(prices) < 26:
        return None, None
    try:
        ema_12 = calculate_ema(prices[-12:], 12)
        ema_26 = calculate_ema(prices[-26:], 26)
        if ema_12 is None or ema_26 is None:
            return None, None
        macd = ema_12 - ema_26
        macd_line = []
        for i in range(26, len(prices)):
            if len(prices[max(0, i-12):i]) >= 12 and len(prices[max(0, i-26):i]) >= 26:
                ema_12_i = calculate_ema(prices[max(0, i-12):i], 12)
                ema_26_i = calculate_ema(prices[max(0, i-26):i], 26)
                if ema_12_i is not None and ema_26_i is not None:
                    macd_line.append(ema_12_i - ema_26_i)
        if len(macd_line) < 9:
            return macd, None
        signal = calculate_ema(macd_line[-9:], 9)
        return macd, signal
    except Exception as e:
        logger.error(f"Lỗi khi tính MACD: {e}")
        return None, None

# --- Tính RSI ---
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
        logger.error(f"Lỗi khi tính RSI: {e}")
        return np.nan

# --- Tính Stochastic Oscillator ---
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
        k_values = [k_value]
        for i in range(1, d_period):
            if len(prices[:-i]) >= k_period:
                low_min_i = min(lows[-k_period-i:-i])
                high_max_i = max(highs[-k_period-i:-i])
                if high_max_i != low_min_i:
                    k = 100 * (prices[-1-i] - low_min_i) / (high_max_i - low_min_i)
                    k_values.append(k)
        d_value = np.mean(k_values) if k_values else None
        return k_value, d_value
    except Exception as e:
        logger.error(f"Lỗi khi tính Stochastic: {e}")
        return None, None

# --- Tính Bollinger Bands ---
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
        logger.error(f"Lỗi khi tính Bollinger Bands: {e}")
        return None, None, None

# --- Phát hiện mô hình nến ---
def detect_candlestick_pattern(highs, lows, opens, closes):
    if len(closes) < 2:
        return None
    try:
        if abs(opens[-1] - closes[-1]) <= (highs[-1] - lows[-1]) * 0.1:
            return "Doji - Tín hiệu đảo chiều tiềm năng"
        return None
    except Exception as e:
        logger.error(f"Lỗi khi phát hiện mô hình nến: {e}")
        return None

# --- Tính Volume Spike ---
def calculate_volume_spike(volumes, period=5, threshold=1.5):
    if len(volumes) < period:
        return False
    try:
        avg_volume = np.mean(volumes[-period:])
        return volumes[-1] > avg_volume * threshold
    except Exception as e:
        logger.error(f"Lỗi khi tính Volume Spike: {e}")
        return False

# --- Phát hiện Breakout ---
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
        logger.error(f"Lỗi khi phát hiện Breakout: {e}")
        return None

# --- Dự đoán giá với Random Forest ---
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
            'low': lows[-30:],
            'atr': [atr_value] * 30 if atr_value is not None else [0] * 30,
            'vwap': [vwap_value] * 30 if vwap_value is not None else [0] * 30
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
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán giá với Random Forest: {e}")
        return None

# --- Phát hiện hỗ trợ và kháng cự ---
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
        logger.error(f"Lỗi khi phát hiện hỗ trợ/kháng cự: {e}")
        return "Ổn định"

# --- Định dạng giá trị ---
def format_value(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{value:.{decimals}f}"
    except Exception as e:
        logger.error(f"Lỗi khi định dạng giá trị: {e}")
        return "N/A"

# --- Lưu dữ liệu vào CSV ---
def save_to_csv(price, trend, win_rate, market_status, chat_id):
    try:
        file_exists = os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Time', 'ChatID', 'Price', 'Trend', 'WinRate', 'MarketStatus'])
            writer.writerow([datetime.now().strftime("%H:%M:%S %d-%m-%Y"), chat_id, format_value(price), trend, format_value(win_rate), market_status])
    except Exception as e:
        logger.error(f"Lỗi khi lưu vào CSV: {e}")

# --- Lệnh Start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"Chào {user.first_name}! Chào mừng bạn đến với @mekiemtien102\n\n"
        "Sử dụng /help để xem danh sách lệnh.\n"
        "Nếu bạn cần hỗ trợ, hãy dùng /cskh."
    )

# --- Lệnh Set Up ---
async def set_up(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    current_time = time.time()
    
    auth_status = "Đã được ủy quyền mặc định" if chat_id == ALLOWED_CHAT_ID else "Chưa được ủy quyền"
    if chat_id in authorized_chats:
        auth_info = authorized_chats[chat_id]
        if auth_info.get("banned", False):
            auth_status = "Đã bị cấm do nhập sai key quá số lần cho phép"
        elif auth_info.get("key_used", False) and current_time - auth_info["timestamp"] < 24 * 3600:
            auth_status = f"Đã được ủy quyền (hết hạn sau {format_value((24 * 3600 - (current_time - auth_info['timestamp'])) / 3600, 1)} giờ)"
    
    api_status = "Chưa cài đặt API Key/Secret"
    if context.user_data.get('api_key') and context.user_data.get('api_secret'):
        api_status = f"API Key: {context.user_data['api_key']}\nAPI Secret: {context.user_data['api_secret']}"
    
    message = (
        f"📋 **Trạng thái thiết lập bot** 📋\n\n"
        f"**Chat ID**: {chat_id}\n"
        f"**Trạng thái ủy quyền**: {auth_status}\n"
        f"**Cài đặt API**:\n{api_status}\n\n"
        f"Nếu chưa ủy quyền, dùng /key <key>.\n"
        f"Để cài đặt API, dùng /settings."
    )
    await update.message.reply_text(message)

# --- Lệnh Help ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_message = (
        "📖 **Hướng dẫn sử dụng @mekiemtien102** 📖\n\n"
        "Danh sách lệnh khả dụng:\n"
        "/start - Khởi động bot và xem giới thiệu\n"
        "/key <key> - Xác thực quyền truy cập cho chat (24 giờ)\n"
        "/signals - Xem tín hiệu giao dịch BTC/USD\n"
        "/settings - Cài đặt API của coincex.io\n"
        "/set_up - Kiểm tra trạng thái thiết lập của bot\n"
        "/help - Hiển thị danh sách lệnh\n"
        "/cskh - Thông tin hỗ trợ khách hàng\n\n"
        "Bot sẽ tự động lưu lịch sử tín hiệu vào file price_history.csv."
    )
    await update.message.reply_text(help_message)

# --- Lệnh CSKH ---
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

# --- Lệnh Signals ---
async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global price_history, volume_history, high_history, low_history, open_history
    chat_id = str(update.effective_chat.id)

    if not await is_allowed_chat(update, context):
        return
    
    async with analysis_lock:
        try:
            latest_price, latest_volume, prices, volumes, highs, lows, opens = get_btc_price_and_volume()
            if latest_price is None:
                await update.message.reply_text("Không thể lấy dữ liệu giá. Vui lòng thử lại.")
                await notify_error(context.application, ALLOWED_CHAT_ID, "Không thể lấy dữ liệu giá từ API Kraken")
                return

            price_history = prices[-MAX_HISTORY:]
            volume_history = volumes[-MAX_HISTORY:]
            high_history = highs[-MAX_HISTORY:]
            low_history = lows[-MAX_HISTORY:]
            open_history = opens[-MAX_HISTORY:]

            if not price_history or not volume_history or not high_history or not low_history or not open_history:
                await update.message.reply_text("Dữ liệu không đủ để phân tích.")
                await notify_error(context.application, ALLOWED_CHAT_ID, "Dữ liệu không đủ để phân tích")
                return

            sma_5 = calculate_sma(price_history, 5)
            sma_20, upper_band, lower_band = calculate_bollinger_bands(price_history)
            macd, signal = calculate_macd(price_history)
            rsi = calculate_rsi(price_history)
            k_value, d_value = calculate_stochastic(price_history, high_history, low_history)
            vwap = calculate_vwap(price_history, volume_history, high_history, low_history)
            atr = calculate_atr(high_history, low_history, price_history)
            fib_382, fib_618, fib_diff = calculate_fibonacci_levels(price_history)
            predicted_price = predict_price_rf(price_history, volume_history, high_history, low_history)
            candlestick_pattern = detect_candlestick_pattern(high_history, low_history, open_history, price_history)
            volume_trend = "TĂNG" if sum(volume_history[-5:]) > sum(volume_history[-10:-5]) else "GIẢM"
            volume_spike = calculate_volume_spike(volume_history)
            breakout = detect_breakout(price_history, high_history, low_history)
            support_resistance_signal = detect_support_resistance(latest_price)

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
                (2.5 if breakout == "Breakout Up" else 0, "Breakout Up")
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
                (2.5 if breakout == "Breakout Down" else 0, "Breakout Down")
            ]

            buy_score = sum(weight for weight, _ in buy_signals)
            sell_score = sum(weight for weight, _ in sell_signals)

            trend = "MUA" if buy_score > sell_score else "BÁN" if sell_score > buy_score else "CHỜ LỆNH"
            total_score = buy_score + sell_score
            win_rate = ((buy_score / total_score) * 100 if trend == "MUA" else (sell_score / total_score) * 100 if trend == "BÁN" else 50) if total_score > 0 else 50

            latest_price_str = format_value(latest_price)
            win_rate_str = format_value(win_rate)

            report = (
                f"🏗️ COINCEX — BTC/USD 🌐\n"
                f"Time: {datetime.now().strftime('%H:%M:%S %d-%m-%Y')}\n"
                f"Lệnh: {trend}\n"
                f"Tỷ lệ thắng: {win_rate_str}%\n"
                f"Giá hiện tại: {latest_price_str} USD"
            )

            try:
                await update.message.reply_text(report)
                logger.info(f"Báo cáo tín hiệu được gửi đến chat {chat_id}")
                save_to_csv(latest_price, trend, win_rate, support_resistance_signal, chat_id)
            except TelegramError as e:
                logger.error(f"Không thể gửi tin nhắn Telegram đến {chat_id}: {e}")
                await update.message.reply_text("Lỗi khi gửi tín hiệu. Vui lòng thử lại.")
                await notify_error(context.application, ALLOWED_CHAT_ID, f"Lỗi khi gửi tín hiệu đến chat {chat_id}: {e}")
        except Exception as e:
            logger.error(f"Lỗi trong lệnh signals: {e}")
            await update.message.reply_text("Lỗi khi xử lý tín hiệu. Vui lòng thử lại.")
            await notify_error(context.application, ALLOWED_CHAT_ID, f"Lỗi trong lệnh signals: {e}")

# --- Lệnh Settings ---
API_KEY, API_SECRET = range(2)

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_allowed_chat(update, context):
        return
    await update.message.reply_text("Vui lòng nhập API Key của bạn từ coincex.io:")
    return API_KEY

async def get_api_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['api_key'] = update.message.text
    await update.message.reply_text("Vui lòng nhập API Secret:")
    return API_SECRET

async def get_api_secret(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['api_secret'] = update.message.text
    try:
        await update.message.reply_text("Cài đặt API thành công! Sử dụng /signals để xem tín hiệu.")
        return ConversationHandler.END
    except TelegramError as e:
        logger.error(f"Không thể gửi xác nhận Telegram đến {update.effective_chat.id}: {e}")
        await notify_error(context.application, ALLOWED_CHAT_ID, f"Lỗi trong cài đặt API: {e}")
        return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Đã hủy cài đặt.")
    return ConversationHandler.END

# --- Xác thực Token Telegram ---
async def validate_token(token: str) -> bool:
    try:
        bot = Bot(token)
        await bot.get_me()
        return True
    except TelegramError as e:
        logger.error(f"Token Telegram không hợp lệ: {e}")
        return False

# --- Hàm chính ---
async def main():
    if not await validate_token(TELEGRAM_TOKEN):
        logger.error("Bot không thể khởi động do token Telegram không hợp lệ.")
        return
    
    try:
        logger.info("Đang tải danh sách chat được ủy quyền...")
        load_authorized_chats()
        logger.info("Khởi tạo bot Telegram...")
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        logger.info("Thêm các trình xử lý lệnh...")
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("set_up", set_up))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("cskh", cskh))
        application.add_handler(CommandHandler("signals", signals))
        application.add_handler(CommandHandler("key", key_command))
        application.add_handler(ConversationHandler(
            entry_points=[CommandHandler("settings", settings)],
            states={
                API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_api_key)],
                API_SECRET: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_api_secret)]
            },
            fallbacks=[CommandHandler("cancel", cancel)]
        ))

        logger.info("Đã thêm các trình xử lý lệnh thành công.")
        logger.info("Đang khởi động bot với polling...")
        await application.initialize()
        await application.start()
        await application.updater.start_polling()

        logger.info("Bot đã khởi động thành công.")
        try:
            await asyncio.Event().wait()  # Giữ bot chạy
        except KeyboardInterrupt:
            logger.info("Nhận tín hiệu tắt, đang dừng bot...")
        finally:
            logger.info("Đang tắt bot...")
            await application.updater.stop()
            await application.stop()
            await application.shutdown()
            logger.info("Bot đã tắt thành công.")
    except Exception as e:
        logger.error(f"Không thể chạy bot: {e}")
        await notify_error(application, ALLOWED_CHAT_ID, f"Không thể khởi động bot: {e}")

# --- Chạy bot ---
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot bị dừng bởi người dùng.")
    except Exception as e:
        logger.error(f"Lỗi khi chạy bot: {e}")
