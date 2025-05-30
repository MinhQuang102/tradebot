import requests
import json
import csv
import os
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
import time
import logging
import websocket
import threading

# --- Setup Logging ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Configuration from Environment Variables ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7608384401:AAHKfX5KlBl5CZTaoKSDwwdATmbY8Z34vRk")
ALLOWED_CHAT_ID = os.getenv("ALLOWED_CHAT_ID", "-1002554202438")
VALID_KEY = os.getenv("VALID_KEY", "10092006")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "af9b016f3f044a6f84453bbe1a526f0b")

if not TELEGRAM_TOKEN or not ALLOWED_CHAT_ID:
    logger.error("TELEGRAM_TOKEN or ALLOWED_CHAT_ID is missing in environment variables.")
    raise ValueError("TELEGRAM_TOKEN or ALLOWED_CHAT_ID is missing in environment variables.")

# --- Constants ---
KRAKEN_OHLC_URL = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1'
COINCEX_WS_URL = 'wss://api.coincex.io:2096/trade-bo/?EIO=4&transport=websocket&sid=rf4Y5WbxdXts4QxDABKK'
HISTORY_FILE = "price_history.csv"
AUTHORIZED_CHATS_FILE = "authorized_chats.json"
MAX_HISTORY = 100
MIN_ANALYSIS_DURATION = 30
MIN_REST_DURATION = 0

# --- Global Variables ---
price_history = []
volume_history = []
high_history = []
low_history = []
open_history = []
coincex_data = {'price': None, 'volume': None, 'high': None, 'low': None, 'open': None}
support_level = None
resistance_level = None
is_analyzing = False
set_up_jobs = {}
authorized_chats = {}
ws_connected = False
ws_thread = None

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
    
    if update.effective_chat.type in ['group', 'supergroup']:
        try:
            member = await update.effective_chat.get_member(user_id)
            if member.status not in ['administrator', 'creator']:
                await update.message.reply_text("Chỉ quản trị viên hoặc người tạo nhóm mới có thể sử dụng các lệnh của bot.")
                logger.warning(f"User {user_id} in chat {chat_id} is not an admin or creator.")
                return False
        except TelegramError as e:
            logger.error(f"Error checking admin status for user {user_id} in chat {chat_id}: {e}")
            await update.message.reply_text("Lỗi khi kiểm tra quyền quản trị viên. Vui lòng thử lại hoặc liên hệ @mekiemtien102.")
            return False
    
    if authorized_chats and chat_id in authorized_chats:
        auth_info = authorized_chats[chat_id]
        if current_time - auth_info["timestamp"] < 24 * 3600:
            return True
        else:
            del authorized_chats[chat_id]
            save_authorized_chats()
    
    try:
        await update.message.reply_text("Đoạn chat này không có quyền sử dụng bot. Vui lòng nhập key bằng lệnh /key <key>. Liên hệ @mekiemtien102 để được hỗ trợ.")
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
                await update.message.reply_text("Đoạn chat này đã bị khóa do nhập key quá số lần cho phép. Liên hệ @mekiemtien102 để được hỗ trợ.")
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
            await update.message.reply_text("Key không hợp lệ. Vui lòng thử lại hoặc liên hệ @mekiemtien102.")
            logger.warning(f"Chat {chat_id} attempted to use invalid key: {provided_key}")
    except Exception as e:
        logger.error(f"Error processing key command in chat {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi xử lý key. Vui lòng thử lại hoặc liên hệ @mekiemtien102.")

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

# --- WebSocket Handlers for CoinCEX ---
def on_message(ws, message):
    global coincex_data, ws_connected
    try:
        if message.startswith('0'):
            logger.info("CoinCEX WebSocket connected")
            ws_connected = True
            ws.send('40')
        elif message.startswith('42'):
            data = json.loads(message[2:])
            event, payload = data
            if event == "trade":
                coincex_data['price'] = float(payload.get('price', coincex_data['price']))
                coincex_data['volume'] = float(payload.get('volume', coincex_data['volume']))
                coincex_data['high'] = float(payload.get('high', coincex_data['high']))
                coincex_data['low'] = float(payload.get('low', coincex_data['low']))
                coincex_data['open'] = float(payload.get('open', coincex_data['open']))
                logger.info(f"CoinCEX data received: {coincex_data}")
    except Exception as e:
        logger.error(f"Error processing CoinCEX WebSocket message: {e}")

def on_error(ws, error):
    global ws_connected
    logger.error(f"CoinCEX WebSocket error: {error}")
    ws_connected = False

def on_close(ws, close_status_code, close_msg):
    global ws_connected
    logger.warning(f"CoinCEX WebSocket closed: {close_status_code} - {close_msg}")
    ws_connected = False

def on_open(ws):
    logger.info("CoinCEX WebSocket opened")
    ws.send('2probe')
    ws.send('5')

# --- Start WebSocket Connection ---
def start_websocket():
    global ws_thread, ws_connected
    try:
        ws = websocket.WebSocketApp(
            COINCEX_WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        logger.info("CoinCEX WebSocket thread started")
    except Exception as e:
        logger.error(f"Error starting CoinCEX WebSocket: {e}")
        ws_connected = False

# --- Get BTC Price and Volume ---
def get_btc_price_and_volume():
    global price_history, volume_history, high_history, low_history, open_history, coincex_data
    try:
        if ws_connected and coincex_data['price'] is not None:
            latest_price = coincex_data['price']
            latest_volume = coincex_data['volume']
            high = coincex_data['high']
            low = coincex_data['low']
            open_price = coincex_data['open']
            
            price_history.append(latest_price)
            volume_history.append(latest_volume)
            high_history.append(high)
            low_history.append(low)
            open_history.append(open_price)
            
            price_history = price_history[-MAX_HISTORY:]
            volume_history = volume_history[-MAX_HISTORY:]
            high_history = high_history[-MAX_HISTORY:]
            low_history = low_history[-MAX_HISTORY:]
            open_history = open_history[-MAX_HISTORY:]
            
            return latest_price, latest_volume, price_history, volume_history, high_history, low_history, open_history
        else:
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

# --- Detect Candlestick Patterns ---
def detect_candlestick_pattern(highs, lows, opens, closes):
    if len(closes) < 2:
        return None
    if abs(opens[-1] - closes[-1]) <= (highs[-1] - lows[-1]) * 0.1:
        return "Doji - Tín hiệu đảo chiều tiềm năng"
    return None

# --- Calculate Volume Spike ---
def calculate_volume_spike(volumes, period=5, threshold=1.5):
    if len(volumes) < period:
        return False
    avg_volume = np.mean(volumes[-period:])
    return volumes[-1] > avg_volume * threshold

# --- Detect Breakout ---
def detect_breakout(prices, highs, lows, period=20):
    if len(prices) < period:
        return None
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
def detect_support_resistance(price):
    global support_level, resistance_level
    if support_level is None or (price is not None and price < support_level * 0.98):
        support_level = price * 0.98 if price is not None else None
    if resistance_level is None or (price is not None and price > resistance_level * 1.02):
        resistance_level = price * 1.02 if price is not None else None
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
- /key <key>: Nhập key để cấp quyền sử dụng bot cho đoạn chat này (ví dụ: /key ABC).
- /set_up <phân_tích> <chờ kết quả> <bắt đầu> <kết thúc>: Tùy chỉnh tự động phân tích (phân tích bao nhiêu giây, chờ kết quả bao nhiêu giây, bắt đầu lúc nào, dừng lúc nào). Ví dụ: /set_up 50 70 13:00 14:00
- /stop: Dừng lệnh /set_up
- /cskh: Liên hệ hỗ trợ qua Telegram
- /help: Hiển thị danh sách các lệnh này
- /analyze: Phân tích ngay lập tức

Gõ /<lệnh> để sử dụng!
Lưu ý: Trong nhóm, chỉ quản trị viên hoặc người tạo nhóm mới có thể sử dụng các lệnh của bot.
"""

# --- Parse Time to Seconds ---
def parse_time_to_seconds(time_str):
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 3600 + minutes * 60
    except ValueError:
        raise ValueError("Định dạng thời gian không hợp lệ. Vui lòng dùng HH:MM (ví dụ: 13:00).")

# --- Get Current Time in Seconds ---
def current_time_in_seconds():
    now = datetime.now()
    return now.hour * 3600 + now.minute * 60 + now.second

# --- Analyze Market ---
async def analyze_market(update: Update, context: ContextTypes.DEFAULT_TYPE, analysis_duration=50):
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
        await update.message.reply_text("Bắt đầu phân tích...")
        logger.info(f"Starting market analysis for chat {chat_id}")
        
        while time.time() - start_time < max(analysis_duration, MIN_ANALYSIS_DURATION):
            if context.bot_data.get('stopping', False):
                logger.info(f"Analysis stopped for chat {chat_id} due to stopping flag.")
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
                logger.debug(f"Collected data: price={latest_price}, volume={latest_volume}")
            else:
                logger.warning("Failed to fetch data in this iteration.")
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error during market analysis: {e}")
        await update.message.reply_text(f"Lỗi trong quá trình phân tích: {str(e)}")
        return
    finally:
        is_analyzing = False
        logger.info(f"Finished market analysis for chat {chat_id}")

    if not temp_prices:
        await update.message.reply_text("Không thể lấy dữ liệu từ BTC hoặc COINCEX. Lỗi 451 - Có thể do mạng bị chặn. Vui lòng thử lại hoặc sử dụng VPN.")
        logger.error("No price data collected during analysis.")
        return

    latest_price = temp_prices[-1]
    price_history = price_history[-MAX_HISTORY:]

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
    candlestick_pattern = detect_candlestick_pattern(high_history, low_history, open_history, price_history)
    volume_trend = "TĂNG" if sum(volume_history[-5:]) > sum(volume_history[-10:-5]) else "GIẢM"
    volume_spike = calculate_volume_spike(volume_history)
    breakout = detect_breakout(price_history, high_history, low_history)
    support_resistance_signal = detect_support_resistance(latest_price)

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
        (1.5 if volume_trend == "TĂNG" else 0, "Volume Up"),
        (1.5 if lower_band is not None and latest_price <= lower_band else 0, "Bollinger Lower"),
        (2.0 if rsi is not None and rsi < 30 else 0, "RSI Oversold"),
        (2.5 if rsi is not None and rsi < 20 else 0, "RSI Strongly Oversold"),
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
        (2.0 if rsi is not None and rsi > 70 else 0, "RSI Overbought"),
        (2.5 if rsi is not None and rsi > 80 else 0, "RSI Strongly Overbought"),
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

    report = f"""
    🏗️ COINCEX — BTC/USD 🌐
    Time: {datetime.now().strftime('%H:%M:%S %d-%m-%Y')}
    Lệnh: {trend}
    Tỷ lệ thắng: {win_rate_str}%
    Giá hiện tại: {latest_price_str} USD
    RSI: {format_value(rsi)}
    Volume Trend: {volume_trend}
    Breakout: {breakout if breakout else 'None'}
    """

    try:
        await update.message.reply_text(report)
        logger.info(f"Analysis report sent to chat {chat_id}")
        save_to_csv(latest_price, trend, win_rate, support_resistance_signal, chat_id)
    except TelegramError as e:
        logger.error(f"Error sending Telegram message to {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi gửi báo cáo. Vui lòng thử lại.")

# --- Analyze Command ---
async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await analyze_market(update, context, analysis_duration=50)

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
        await update.message.reply_text("Vui lòng cung cấp đúng định dạng: /set_up <phân_tích> <chờ kết quả> <bắt đầu> <kết thúc>\nVí dụ: /set_up 50 70 13:00 14:00")
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
        await update.message.reply_text(f"Lỗi: {str(e)}. Vui lòng kiểm tra định dạng thời gian (HH:MM).")
    except Exception as e:
        await update.message.reply_text(f"Lỗi không xác định: {str(e)}")

# --- Stop Command ---
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_allowed_chat(update, context):
        return
    chat_id = str(update.message.chat_id)
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
    support_message = "Cần hỗ trợ? Liên hệ qua Telegram: @mekiemtien102"
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

# --- Main Function ---
async def main():
    try:
        logger.info("Loading authorized chats...")
        load_authorized_chats()
        logger.info("Starting CoinCEX WebSocket...")
        start_websocket()
        logger.info("Initializing Telegram bot...")
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        logger.info("Bot initialized. Adding handlers...")

        app.add_handler(CommandHandler("set_up", set_up_command))
        app.add_handler(CommandHandler("stop", stop_command))
        app.add_handler(CommandHandler("cskh", cskh_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("key", key_command))
        app.add_handler(CommandHandler("analyze", analyze_command))
        logger.info("Handlers added.")

        port = int(os.getenv("PORT", 8443))
        webhook_url = f"https://tradebot-r6x3.onrender.com/{TELEGRAM_TOKEN}"
        logger.info(f"Setting webhook with URL: {webhook_url}")
        await app.bot.set_webhook(url=webhook_url)
        
        await app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=TELEGRAM_TOKEN,
            webhook_url=webhook_url
        )
        logger.info("Bot is running with webhook.")
    except TelegramError as e:
        logger.error(f"Telegram bot error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
