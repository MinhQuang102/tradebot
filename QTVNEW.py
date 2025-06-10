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
from telegram.error import TelegramError, RetryAfter, TimedOut
import time
import logging
from aiohttp import web
import random
from typing import List, Optional, Tuple
import sys

# --- Setup Logging ---
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Load Configuration ---
CONFIG_FILE = "coincex.json"

def load_config() -> dict:
    try:
        # Prefer environment variables
        telegram_tokens = os.getenv('TELEGRAM_TOKENS', '').split(',')
        news_api_key = os.getenv('NEWS_API_KEY', '')
        
        if telegram_tokens and news_api_key and telegram_tokens[0]:
            return {
                "TELEGRAM_TOKENS": [t.strip() for t in telegram_tokens if t.strip()],
                "NEWS_API_KEY": news_api_key
            }
        
        # Fallback to config file
        config = {"TELEGRAM_TOKENS": [], "NEWS_API_KEY": ""}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if not isinstance(config.get("TELEGRAM_TOKENS"), list):
                    config["TELEGRAM_TOKENS"] = []
                if not isinstance(config.get("NEWS_API_KEY"), str):
                    config["NEWS_API_KEY"] = ""
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {CONFIG_FILE}: {e}. Using default config.")
            except Exception as e:
                logger.error(f"Error reading {CONFIG_FILE}: {e}. Using default config.")
        
        return config
    except Exception as e:
        logger.error(f"Configuration error: {e}. Using default config.")
        return {"TELEGRAM_TOKENS": [], "NEWS_API_KEY": ""}

try:
    config = load_config()
    TELEGRAM_TOKENS = config.get('TELEGRAM_TOKENS', [])
    NEWS_API_KEY = config.get('NEWS_API_KEY', '')
    if not TELEGRAM_TOKENS or not NEWS_API_KEY:
        raise ValueError("Missing TELEGRAM_TOKENS or NEWS_API_KEY in configuration")
except Exception as e:
    logger.error(f"Configuration error: {e}")
    sys.exit(1)

# --- Constants ---
KRAKEN_OHLC_URL = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1'
HISTORY_FILE = "price_history.csv"
MAX_HISTORY = 30
MIN_ANALYSIS_DURATION = 5
MIN_RESTART_DELAY = 5
REQUEST_TIMEOUT = 5

# --- Global Variables ---
price_history: List[float] = []
volume_history: List[float] = []
high_history: List[float] = []
low_history: List[float] = []
open_history: List[float] = []
support_level: Optional[float] = None
resistance_level: Optional[float] = None
is_analyzing = False

# --- Get BTC Price and Volume ---
async def get_btc_price_and_volume() -> Tuple[Optional[float], Optional[float], List[float], List[float], List[float], List[float], List[float]]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(KRAKEN_OHLC_URL, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                data = await response.json()
        
        if 'error' in data and data['error']:
            logger.warning(f"Kraken API error: {data['error']}")
            return None, None, [], [], [], [], []
        
        klines = data['result']['XXBTZUSD'][:MAX_HISTORY]
        if not klines or len(klines) < MAX_HISTORY:
            logger.warning(f"Kraken data insufficient: {len(klines)} candles received")
            return None, None, [], [], [], [], []
        
        prices = [float(candle[4]) for candle in klines]
        volumes = [float(candle[5]) for candle in klines]
        highs = [float(candle[2]) for candle in klines]
        lows = [float(candle[3]) for candle in klines]
        opens = [float(candle[1]) for candle in klines]
        latest_price = prices[-1]
        latest_volume = volumes[-1]
        
        return latest_price, latest_volume, prices, volumes, highs, lows, opens
    except Exception as e:
        logger.error(f"Error in get_btc_price_and_volume: {e}")
        return None, None, [], [], [], [], []

# --- Calculate VWAP ---
def calculate_vwap(prices: List[float], volumes: List[float], highs: List[float], lows: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period or len(volumes) < period:
        return None
    typical_prices = [(highs[i] + lows[i] + prices[i]) / 3 for i in range(-period, 0)]
    vwap = sum(typical_prices[i] * volumes[i] for i in range(-period, 0)) / sum(volumes[-period:])
    return vwap

# --- Calculate ATR ---
def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
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
def calculate_fibonacci_levels(prices: List[float], period: int = 100) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < period:
        return None, None, None
    high = max(prices[-period:])
    low = min(prices[-period:])
    diff = high - low
    fib_382 = high - diff * 0.382
    fib_618 = high - diff * 0.618
    return fib_382, fib_618, diff

# --- Calculate SMA ---
def calculate_sma(prices: List[float], period: int = 5) -> Optional[float]:
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

# --- Calculate EMA ---
def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period or not prices:
        return None
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema

# --- Calculate MACD ---
def calculate_macd(prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
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
def calculate_rsi(prices: List[float], period: int = 14) -> float:
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
def calculate_stochastic(prices: List[float], highs: List[float], lows: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
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
def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < period:
        return None, None, None
    sma = sum(prices[-period:]) / period
    std = np.std(prices[-period:])
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

# --- Detect Candlestick Patterns ---
def detect_candlestick_pattern(highs: List[float], lows: List[float], opens: List[float], closes: List[float]) -> Optional[str]:
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
def detect_chart_patterns(prices: List[float], highs: List[float], lows: List[float], period: int = 50) -> Optional[str]:
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
    
    return None

# --- Calculate Volume Spike ---
def calculate_volume_spike(volumes: List[float], period: int = 5, threshold: float = 1.5) -> bool:
    if len(volumes) < period:
        return False
    avg_volume = np.mean(volumes[-period:])
    return volumes[-1] > avg_volume * threshold

# --- Detect Breakout ---
def detect_breakout(prices: List[float], highs: List[float], lows: List[float], volumes: List[float], period: int = 20) -> Tuple[Optional[str], bool]:
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
def analyze_price_behavior(prices: List[float], highs: List[float], lows: List[float], opens: List[float], closes: List[float]) -> Optional[str]:
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
def check_timing_entry() -> str:
    current_hour = datetime.now().hour
    if 8 <= current_hour <= 16:
        return "Good Timing - Phiên Âu/Mỹ"
    return "Caution - Tránh giờ tin tức hoặc phiên thấp điểm"

# --- Predict Price with Random Forest ---
def predict_price_rf(prices: List[float], volumes: List[float], highs: List[float], lows: List[float]) -> Optional[float]:
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
async def get_economic_news() -> List[str]:
    NEWS_API_URL = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={NEWS_API_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(NEWS_API_URL, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
        if 'articles' in data:
            return [article['title'] for article in data['articles'][:3]]
        return []
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

# --- Get Economic Calendar ---
def get_economic_calendar() -> List[str]:
    try:
        return ["USD Non-Farm Payrolls at 14:30", "FOMC Interest Rate Decision at 20:00"]
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {e}")
        return []

# --- Calculate Risk Management ---
def calculate_risk_management(capital: float, risk_percentage: float = 0.02, atr: Optional[float] = None) -> float:
    if atr is None:
        atr = calculate_atr(high_history, low_history, price_history)
    return capital * risk_percentage / atr if atr else capital * risk_percentage

# --- Martingale Strategy ---
def martingale_strategy(last_bet: float, win: bool = False) -> float:
    return last_bet * 2 if not win else last_bet

# --- Detect Support and Resistance ---
def detect_support_resistance(price: Optional[float]) -> str:
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
def format_value(value: Optional[float], decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"

# --- Save Data to CSV ---
def save_to_csv(price: float, trend: str, win_rate: float, market_status: str, chat_id: str):
    try:
        file_exists = os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Time', 'ChatID', 'Price', 'Trend', 'WinRate', 'MarketStatus'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), chat_id, price, trend, win_rate, market_status])
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

# --- Get Help Text ---
def get_help_text() -> str:
    return """
Danh sách các lệnh hỗ trợ:
- /give: Lấy kết quả phân tích thị trường sau 5 giây.
- /cskh: Liên hệ hỗ trợ qua Telegram.
- /help: Hiển thị danh sách các lệnh này.

Gõ /<lệnh> để sử dụng!
"""

# --- Analyze Market ---
async def analyze_market(update: Update, context: ContextTypes.DEFAULT_TYPE, analysis_duration: float = 5) -> None:
    global price_history, volume_history, high_history, low_history, open_history, is_analyzing
    chat_id = str(update.effective_chat.id)
    
    if is_analyzing:
        try:
            await update.message.reply_text("Phân tích đang diễn ra, vui lòng đợi.")
            return
        except Exception as e:
            logger.error(f"Error sending busy message: {e}")
            return
    
    is_analyzing = True
    try:
        start_time = time.time()
        temp_data = {
            'prices': [],
            'volumes': [],
            'highs': [],
            'lows': [],
            'opens': []
        }
        
        while time.time() - start_time < max(analysis_duration, MIN_ANALYSIS_DURATION):
            if context.bot_data.get('stopping', False):
                break
            latest_price, latest_volume, prices, volumes, highs, lows, opens = await get_btc_price_and_volume()
            if latest_price is not None:
                temp_data['prices'].append(latest_price)
                temp_data['volumes'].append(latest_volume)
                temp_data['highs'].append(highs[-1])
                temp_data['lows'].append(lows[-1])
                temp_data['opens'].append(opens[-1])
                price_history = prices[-MAX_HISTORY:]
                volume_history = volumes[-MAX_HISTORY:]
                high_history = highs[-MAX_HISTORY:]
                low_history = lows[-MAX_HISTORY:]
                open_history = opens[-MAX_HISTORY:]
            else:
                logger.warning("Failed to fetch data in this iteration")
            await asyncio.sleep(1)
        
        if not temp_data['prices']:
            await update.message.reply_text("Không thể lấy dữ liệu từ Kraken. Vui lòng thử lại.")
            return
        
        latest_price = temp_data['prices'][-1]
        price_history = price_history[-MAX_HISTORY:]
        
        # Technical analysis
        sma_5 = calculate_sma(price_history)
        ema_9 = calculate_ema(price_history, 9)
        ema_21 = calculate_ema(price_history, 21)
        sma_20, upper_band, lower_band = calculate_bollinger_bands(price_history)
        macd, signal = calculate_macd(price_history)
        rsi = calculate_rsi(price_history)
        k_value, d_value = calculate_stochastic(price_history, high_history, low_history)
        vwap = calculate_vwap(price_history, volume_history, high_history, low_history)
        atr = calculate_atr(high_history, low_history, price_history)
        fib_382, fib_618, _ = calculate_fibonacci_levels(price_history)
        predicted_price = predict_price_rf(price_history, volume_history, high_history, low_history)
        candlestick_pattern = detect_candlestick_pattern(high_history, low_history, open_history, price_history)
        chart_pattern = detect_chart_patterns(price_history, high_history, low_history)
        volume_trend = "TĂNG" if sum(volume_history[-5:]) > sum(volume_history[-10:-5]) else "GIẢM"
        volume_spike = calculate_volume_spike(volume_history)
        breakout, is_true_breakout = detect_breakout(price_history, high_history, low_history, volume_history)
        support_resistance_signal = detect_support_resistance(latest_price)
        price_behavior = analyze_price_behavior(price_history, high_history, low_history, open_history, price_history)
        timing_entry = check_timing_entry()
        
        # Data analysis
        news = await get_economic_news()
        calendar = get_economic_calendar()
        
        # Risk management
        capital = 1000
        position_size = calculate_risk_management(capital)
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
            (1.5 if chart_pattern in ["Double Bottom - Tín hiệu đảo chiều tăng"] else 0, chart_pattern),
            (2.0 if volume_spike else 0, "Volume Spike Buy"),
            (2.5 if breakout == "Breakout Up - Tín hiệu tăng mạnh" and is_true_breakout else 0, "Breakout Up"),
            (1.5 if price_behavior == "Accumulation - Giá tích lũy" else 0, "Price Accumulation"),
            (1.2 if timing_entry == "Good Timing - Phiên Âu/Mỹ" else 0, "Good Timing")
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
            (1.5 if chart_pattern in ["Double Top - Tín hiệu đảo chiều giảm", "Head and Shoulders - Tín hiệu đảo chiều giảm"] else 0, chart_pattern),
            (2.0 if volume_spike else 0, "Volume Spike Sell"),
            (2.5 if breakout == "Breakout Down - Tín hiệu giảm mạnh" and is_true_breakout else 0, "Breakout Down"),
            (1.5 if price_behavior == "Weak Uptrend - Lực mua yếu" else 0, "Weak Uptrend"),
            (1.0 if timing_entry == "Caution - Tránh giờ tin tức hoặc phiên thấp điểm" else 0, "Poor Timing")
        ]
        
        buy_score = sum(weight for weight, _ in buy_signals)
        sell_score = sum(weight for weight, _ in sell_signals)
        
        trend = "MUA" if buy_score > sell_score else "BÁN" if sell_score > buy_score else "CHỜ LỆNH"
        total_score = buy_score + sell_score
        win_rate = ((buy_score / total_score) * 100 if trend == "MUA" else (sell_score / total_score) * 100 if trend == "BÁN" else 50) if total_score > 0 else 50
        
        latest_price_str = format_value(latest_price)
        win_rate_str = format_value(win_rate)
        market_status = f"{support_resistance_signal}\n"
        if candlestick_pattern:
            market_status += f"Nến: {candlestick_pattern}\n"
        if chart_pattern:
            market_status += f"Mô hình: {chart_pattern}\n"
        if breakout:
            market_status += f"Breakout: {breakout}\n"
        if price_behavior:
            market_status += f"Hành vi giá: {price_behavior}\n"
        market_status += f"Thời điểm: {timing_entry}"
        
        report = f"""
📈 **Coincex - BTC/USD** 📈
🕒 **Thời gian**: {datetime.now().strftime('%H:%M:%S %d-%m-%Y')}
💹 **Giá hiện tại**: {latest_price_str} USD
🚨 **Tín hiệu**: {trend}
🎯 **Tỷ lệ thắng**: {win_rate_str}%
📋 **Trạng thái**: {market_status}
"""
        
        for token in TELEGRAM_TOKENS:
            try:
                await context.bot.send_message(chat_id=chat_id, text=report)
                logger.info(f"Analysis report sent to chat {chat_id} with token {token[:8]}...")
                save_to_csv(latest_price, trend, win_rate, market_status, chat_id)
            except (RetryAfter, TimedOut) as e:
                logger.warning(f"Telegram rate limit hit for {chat_id}: {e}. Waiting {e.retry_after} seconds.")
                await asyncio.sleep(e.retry_after + 1)
            except TelegramError as e:
                logger.error(f"Error sending to {chat_id} with token {token[:8]}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending to {chat_id}: {e}")
    except Exception as e:
        logger.error(f"Error in analyze_market for {chat_id}: {e}")
        await update.message.reply_text("Lỗi phân tích thị trường. Vui lòng thử lại.")
    finally:
        is_analyzing = False

# --- Give Command ---
async def give_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    try:
        await update.message.reply_text("Đang phân tích thị trường, vui lòng đợi 5 giây...")
        wait_time = random.uniform(5, 10)
        await asyncio.sleep(wait_time)
        await analyze_market(update, context, analysis_duration=5)
    except Exception as e:
        logger.error(f"Error in give_command for chat {chat_id}: {e}")
        await update.message.reply_text("Lỗi khi thực hiện lệnh /give. Vui lòng thử lại hoặc liên hệ @mekiemtienlamnha.")

# --- CSKH Command ---
async def cskh_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    support_message = "Cần hỗ trợ? Liên hệ qua Telegram: @mekiemtienlamnha"
    try:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=support_message)
    except Exception as e:
        logger.error(f"Error in cskh_command: {e}")
        await update.message.reply_text("Lỗi khi gửi thông tin hỗ trợ. Vui lòng thử lại.")

# --- Help Command ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = get_help_text()
    try:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)
    except Exception as e:
        logger.error(f"Error in help_command: {e}")
        await update.message.reply_text("Lỗi khi gửi danh sách lệnh. Vui lòng thử lại.")

# --- Web Server Handler ---
async def health_check(request: web.Request) -> web.Response:
    return web.json_response({"status": "healthy"})

# --- Start Bot ---
async def start_bot(token: str) -> Application:
    try:
        application = Application.builder().token(token).build()
        application.add_handler(CommandHandler("give", give_command))
        application.add_handler(CommandHandler("cskh", cskh_command))
        application.add_handler(CommandHandler("help", help_command))
        logger.info(f"Bot initialized with token {token[:8]}...")
        return application
    except Exception as e:
        logger.error(f"Failed to initialize bot with token {token[:8]}: {e}")
        raise

# --- Run Web Server and Bots ---
async def run_services():
    app = web.Application()
    app.router.add_get('/health', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv('PORT', 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Web server started on port {port}")
    
    bot_applications = []
    for token in TELEGRAM_TOKENS:
        try:
            application = await start_bot(token)
            bot_applications.append(application)
            await application.initialize()
            await application.updater.start_polling(
                drop_pending_updates=True,
                poll_interval=1.0,
                timeout=10,
                read_timeout=20,
                write_timeout=20,
                connect_timeout=20
            )
            await application.start()
            logger.info(f"Bot started polling with token {token[:8]}...")
        except Exception as e:
            logger.error(f"Error starting bot with token {token[:8]}: {e}")
            continue
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for application in bot_applications:
            await application.stop()
            await application.updater.stop()
        await runner.cleanup()

# --- Main Entry ---
def main():
    try:
        asyncio.run(run_services())
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        time.sleep(MIN_RESTART_DELAY)
        sys.exit(1)

if __name__ == "__main__":
    main()
