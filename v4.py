"""        
GaryBot Pro - Advanced Multi-Timeframe Trading Bot (v4.1 - Fixed)
Bot de trading avanzado con análisis multi-timeframe y escaneo cada 5 minutos

Características principales:
• Análisis multi-timeframe (5m, 15m, 1h, 4h, 1d)
• 10+ indicadores técnicos avanzados
• Sistema de confianza inteligente
• Multi-usuario automático
• Risk management optimizado
• Escaneo cada 5 minutos
• Markdown parsing mejorado

Requisitos:
pip install python-telegram-bot pandas requests python-dotenv binance-python ta-lib numpy
"""

import os
import json
import time
import asyncio
import sqlite3
import signal
import sys
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
from telegram.request import HTTPXRequest
from telegram.error import NetworkError, TimedOut, RetryAfter
from telegram.constants import ParseMode

# Análisis técnico avanzado
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib no disponible, usando implementaciones propias")

# Binance API
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ python-binance no disponible")

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# URLs de API optimizadas
BINANCE_24HR_API = 'https://api.binance.com/api/v3/ticker/24hr'
BINANCE_KLINES_API = 'https://api.binance.com/api/v3/klines'
BINANCE_TICKER_API = 'https://api.binance.com/api/v3/ticker/price'

# Archivos de configuración
USERS_DB = "users.db"
PREFS_FILE = "bot_prefs.json"
SIGNALS_LOG = "signals_history.json"

# Configuración de escaneo
SCAN_INTERVAL = 300  # 5 minutos (300 segundos)

# ============================================================================
# CLASES Y ESTRUCTURAS DE DATOS
# ============================================================================

class TimeFrame(Enum):
    M5 = "5m"
    M15 = "15m" 
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT" 
    SCALP_LONG = "SCALP_LONG"
    SCALP_SHORT = "SCALP_SHORT"
    SWING_LONG = "SWING_LONG"
    SWING_SHORT = "SWING_SHORT"

@dataclass
class TechnicalData:
    """Datos técnicos para un timeframe específico"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    adx: Optional[float] = None
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ema_8: Optional[float] = None
    ema_13: Optional[float] = None
    ema_21: Optional[float] = None
    ema_55: Optional[float] = None
    sar: Optional[float] = None
    williams_r: Optional[float] = None
    vwap: Optional[float] = None
    volume_spike: bool = False

@dataclass
class Signal:
    """Estructura de una señal de trading"""
    symbol: str
    signal_type: SignalType
    timeframe: TimeFrame
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reason: str
    timestamp: datetime
    indicators: Dict[str, float]

@dataclass
class User:
    """Usuario del bot"""
    user_id: int
    username: str
    first_name: str
    join_date: datetime
    is_active: bool = True
    signal_count: int = 0
    last_activity: Optional[datetime] = None

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================

# Estado del bot
app = None
running = False
scan_task = None
connection_retry_count = 0
max_connection_retries = 5

# Configuración
selected_symbols: List[str] = []
macro_alerts = True
min_confidence = 70
active_timeframes = [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1, TimeFrame.H4]

# Cache y datos
price_cache: Dict[str, Tuple[float, float]] = {}
data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
users_cache: Dict[int, User] = {}
dominance_cache: Dict[str, Tuple[float, float]] = {}

# Timeframe weights para confluencia
TIMEFRAME_WEIGHTS = {
    TimeFrame.M5: 1.0,   # Scalping
    TimeFrame.M15: 1.5,  # Short term
    TimeFrame.H1: 2.0,   # Medium term
    TimeFrame.H4: 2.5,   # Swing
    TimeFrame.D1: 3.0    # Long term
}

# Configuración de cache
CACHE_TIMEOUT = 30  # segundos para precios
DATA_CACHE_TIMEOUT = 60  # segundos para datos OHLCV
DOM_CACHE_TIMEOUT = 300  # 5 minutos para dominancia

# API Keys (to be loaded from .env)
TELEGRAM_BOT_TOKEN = None
BINANCE_API_KEY = None
BINANCE_API_SECRET = None

# ============================================================================
# UTILIDADES DE TEXTO SEGURO PARA TELEGRAM
# ============================================================================

def escape_markdown_v2(text: str) -> str:
    """Escapa caracteres especiales para MarkdownV2"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    return text

def safe_markdown_text(text: str) -> str:
    """Crea texto seguro para Telegram sin formateo especial"""
    text = re.sub(r'[*_`\[\]()~>#+=|{}.!-]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def create_safe_message(symbol: str, signal_type: str, timeframe: str, confidence: float, 
                       entry_price: float, stop_loss: float, take_profit: float, 
                       risk_reward: float, reason: str) -> str:
    """Crea un mensaje seguro sin caracteres especiales problemáticos"""
    
    signal_icons = {
        "LONG": "🟢 LONG",
        "SHORT": "🔴 SHORT", 
        "SCALP_LONG": "⚡🟢 SCALP LONG",
        "SCALP_SHORT": "⚡🔴 SCALP SHORT",
        "SWING_LONG": "📈🟢 SWING LONG",
        "SWING_SHORT": "📉🔴 SWING SHORT"
    }
    
    icon_text = signal_icons.get(signal_type, f"🎯 {signal_type}")
    
    def safe_format_price(price: float) -> str:
        if price >= 1000:
            return f"{price:,.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        elif price >= 0.001:
            return f"{price:.6f}"
        else:
            return f"{price:.8f}"
    
    if signal_type in ["LONG", "SCALP_LONG", "SWING_LONG"]:
        sl_pct = ((entry_price - stop_loss) / entry_price) * 100
        tp_pct = ((take_profit - entry_price) / entry_price) * 100
    else:
        sl_pct = ((stop_loss - entry_price) / entry_price) * 100
        tp_pct = ((entry_price - take_profit) / entry_price) * 100
    
    safe_reason = safe_markdown_text(reason)
    
    message = f"""🚨 SEÑAL AUTOMÁTICA

{icon_text}
{symbol} | {timeframe} | {confidence:.1f}%

💰 Entry: ${safe_format_price(entry_price)}
🛑 Stop Loss: ${safe_format_price(stop_loss)} ({sl_pct:.2f}%)
🎯 Take Profit: ${safe_format_price(take_profit)} ({tp_pct:.2f}%)

📊 Risk/Reward: 1:{risk_reward:.1f}
🔍 Razón: {safe_reason}

⏰ {datetime.now().strftime('%H:%M:%S')}"""
    
    return message.strip()

# ============================================================================
# CONFIGURACIÓN ROBUSTA DE RED
# ============================================================================

def create_telegram_request() -> HTTPXRequest:
    """Crea un objeto de request con configuración robusta para timeouts"""
    return HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=10.0,
    )

async def test_connection():
    """Prueba la conexión a Telegram"""
    if not TELEGRAM_BOT_TOKEN:
        return False, "Token no configurado"
    
    try:
        import httpx
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe")
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    bot_info = data.get("result", {})
                    return True, f"Bot conectado: {bot_info.get('first_name', 'Unknown')}"
                else:
                    return False, f"API Error: {data.get('description', 'Unknown error')}"
            else:
                return False, f"HTTP Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# ============================================================================
# MANEJADOR DE SEÑALES PARA SHUTDOWN LIMPIO
# ============================================================================

def signal_handler(signum, frame):
    """Manejador de señales para shutdown limpio"""
    global running
    print("\n🛑 Recibida señal de interrupción...")
    running = False

# Registrar manejadores de señales
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# BASE DE DATOS Y PERSISTENCIA
# ============================================================================

def init_database():
    """Inicializa la base de datos SQLite para usuarios"""
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            join_date TEXT,
            is_active INTEGER DEFAULT 1,
            signal_count INTEGER DEFAULT 0,
            last_activity TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            signal_type TEXT,
            timeframe TEXT,
            confidence REAL,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            risk_reward REAL,
            timestamp TEXT,
            user_count INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(user_id: int, username: str = None, first_name: str = None):
    """Añade un usuario nuevo a la base de datos"""
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    cursor.execute('''
        INSERT OR REPLACE INTO users 
        (user_id, username, first_name, join_date, is_active, last_activity)
        VALUES (?, ?, ?, ?, 1, ?)
    ''', (user_id, username or "", first_name or "", now, now))
    
    conn.commit()
    conn.close()
    
    users_cache[user_id] = User(
        user_id=user_id,
        username=username or "",
        first_name=first_name or "",
        join_date=datetime.utcnow(),
        is_active=True,
        last_activity=datetime.utcnow()
    )

def get_active_users() -> List[User]:
    """Obtiene todos los usuarios activos"""
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE is_active = 1')
    rows = cursor.fetchall()
    conn.close()
    
    users = []
    for row in rows:
        user = User(
            user_id=row[0],
            username=row[1],
            first_name=row[2], 
            join_date=datetime.fromisoformat(row[3]),
            is_active=bool(row[4]),
            signal_count=row[5],
            last_activity=datetime.fromisoformat(row[6]) if row[6] else None
        )
        users.append(user)
        users_cache[user.user_id] = user
    
    return users

def update_user_activity(user_id: int):
    """Actualiza la última actividad del usuario"""
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    cursor.execute('UPDATE users SET last_activity = ? WHERE user_id = ?', (now, user_id))
    conn.commit()
    conn.close()
    
    if user_id in users_cache:
        users_cache[user_id].last_activity = datetime.utcnow()

def save_signal_to_history(signal: Signal, user_count: int):
    """Guarda una señal en el historial"""
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO signals_history 
        (symbol, signal_type, timeframe, confidence, entry_price, stop_loss, 
         take_profit, risk_reward, timestamp, user_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        signal.symbol,
        signal.signal_type.value,
        signal.timeframe.value, 
        signal.confidence,
        signal.entry_price,
        signal.stop_loss,
        signal.take_profit,
        signal.risk_reward,
        signal.timestamp.isoformat(),
        user_count
    ))
    
    conn.commit()
    conn.close()

def load_preferences():
    """Carga preferencias del archivo JSON"""
    global selected_symbols, macro_alerts, min_confidence, active_timeframes
    
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, 'r') as f:
                data = json.load(f)
            
            selected_symbols = data.get('symbols', [])
            macro_alerts = data.get('macro_alerts', True)
            min_confidence = data.get('min_confidence', 70)
            
            tf_strings = data.get('active_timeframes', ['5m', '15m', '1h', '4h'])
            active_timeframes = [TimeFrame(tf) for tf in tf_strings if tf in [t.value for t in TimeFrame]]
            
        except Exception as e:
            print(f"Error cargando preferencias: {e}")
            reset_preferences()

def save_preferences():
    """Guarda preferencias en archivo JSON"""
    data = {
        'symbols': selected_symbols,
        'macro_alerts': macro_alerts,
        'min_confidence': min_confidence,
        'active_timeframes': [tf.value for tf in active_timeframes]
    }
    
    try:
        with open(PREFS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error guardando preferencias: {e}")

def reset_preferences():
    """Resetea preferencias a valores por defecto"""
    global selected_symbols, macro_alerts, min_confidence, active_timeframes
    selected_symbols = []
    macro_alerts = True
    min_confidence = 70
    active_timeframes = [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1, TimeFrame.H4]

# ============================================================================
# FUNCIONES DE API Y DATOS CON RETRY LOGIC
# ============================================================================

async def make_request_with_retry(url: str, params: dict = None, max_retries: int = 3, timeout: int = 10) -> Optional[requests.Response]:
    """Hace requests con lógica de retry"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout en intento {attempt + 1}/{max_retries} para {url}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error de red en intento {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    
    return None

def get_top_symbols_by_volume(limit: int = 50) -> List[str]:
    """Obtiene los top símbolos por volumen de trading"""
    try:
        response = requests.get(BINANCE_24HR_API, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        usdt_pairs = [item for item in data 
                     if item['symbol'].endswith('USDT') and 
                     float(item['quoteVolume']) > 10000000]
        
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
        return [item['symbol'] for item in sorted_pairs[:limit]]
        
    except Exception as e:
        print(f"Error obteniendo top symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", 
                "XRPUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT"]

def is_valid_trading_symbol(symbol: str) -> bool:
    """Valida si un símbolo es válido para trading"""
    if not symbol or len(symbol) < 6:
        return False
    
    valid_quotes = ['USDT', 'BUSD', 'USDC']
    return any(symbol.endswith(quote) for quote in valid_quotes)

async def get_current_price(symbol: str) -> Optional[float]:
    """Obtiene precio actual con cache y retry"""
    current_time = time.time()
    
    if symbol in price_cache:
        cached_price, cached_time = price_cache[symbol]
        if current_time - cached_time < CACHE_TIMEOUT:
            return cached_price
    
    response = await make_request_with_retry(f"{BINANCE_TICKER_API}?symbol={symbol}", timeout=8)
    if response:
        try:
            data = response.json()
            price = float(data['price'])
            
            price_cache[symbol] = (price, current_time)
            return price
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error procesando precio {symbol}: {e}")
    
    return None

async def get_klines_data(symbol: str, timeframe: TimeFrame, limit: int = 200) -> Optional[pd.DataFrame]:
    """Obtiene datos OHLCV con cache por timeframe y retry"""
    current_time = time.time()
    
    cache_key = f"{symbol}_{timeframe.value}"
    if symbol in data_cache and timeframe.value in data_cache[symbol]:
        cached_df = data_cache[symbol][timeframe.value]
        if hasattr(cached_df, '_cache_time') and current_time - cached_df._cache_time < DATA_CACHE_TIMEOUT:
            return cached_df
    
    params = {
        'symbol': symbol,
        'interval': timeframe.value,
        'limit': limit
    }
    
    response = await make_request_with_retry(BINANCE_KLINES_API, params=params, timeout=15)
    if response:
        try:
            klines = response.json()
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df._cache_time = current_time
            
            if symbol not in data_cache:
                data_cache[symbol] = {}
            data_cache[symbol][timeframe.value] = df
            
            return df
            
        except Exception as e:
            print(f"Error procesando klines {symbol} {timeframe.value}: {e}")
    
    return None

# ============================================================================
# INDICADORES TÉCNICOS
# ============================================================================

def calculate_rsi(close_prices: np.array, period: int = 14) -> Optional[float]:
    """RSI optimizado con suavizado de Wilder"""
    if TALIB_AVAILABLE:
        try:
            rsi_values = talib.RSI(close_prices, timeperiod=period)
            return rsi_values[-1] if not np.isnan(rsi_values[-1]) else None
        except:
            pass
    
    if len(close_prices) < period + 1:
        return None
    
    try:
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        alpha = 1.0 / period
        avg_gain = pd.Series(gains).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        avg_loss = pd.Series(losses).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    except Exception as e:
        print(f"Error calculando RSI: {e}")
        return None

async def analyze_symbol_technical(symbol: str, timeframe: TimeFrame) -> Optional[TechnicalData]:
    """Análisis técnico simplificado para un símbolo y timeframe"""
    try:
        df = await get_klines_data(symbol, timeframe, 200)
        if df is None or len(df) < 50:
            return None
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        tech_data = TechnicalData()
        
        tech_data.rsi = calculate_rsi(close)
        
        if len(close) >= 21:
            tech_data.ema_8 = pd.Series(close).ewm(span=8).mean().iloc[-1]
            tech_data.ema_21 = pd.Series(close).ewm(span=21).mean().iloc[-1]
            if len(close) >= 55:
                tech_data.ema_55 = pd.Series(close).ewm(span=55).mean().iloc[-1]
        
        if len(close) >= 20:
            sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]
            std_20 = pd.Series(close).rolling(20).std().iloc[-1]
            tech_data.bb_middle = sma_20
            tech_data.bb_upper = sma_20 + (2 * std_20)
            tech_data.bb_lower = sma_20 - (2 * std_20)
        
        if len(volume) >= 20:
            avg_volume = np.mean(volume[-20:-1])
            current_volume = volume[-1]
            tech_data.volume_spike = current_volume > (avg_volume * 1.5)
        
        return tech_data
        
    except Exception as e:
        print(f"Error en análisis técnico {symbol} {timeframe.value}: {e}")
        return None

# ============================================================================
# SISTEMA DE SEÑALES
# ============================================================================

async def calculate_signal_confidence(tech_data: TechnicalData, symbol: str, timeframe: TimeFrame, macro_context: Dict) -> Tuple[Optional[SignalType], float, str]:
    """Cálculo de confianza de señal mejorado"""
    
    if not tech_data:
        return None, 0.0, "No hay datos técnicos"
    
    bullish_signals = 0
    bearish_signals = 0
    total_indicators = 0
    reasons = []
    
    current_price = await get_current_price(symbol)
    if not current_price:
        return None, 0.0, "No se pudo obtener precio actual"
    
    # RSI Analysis
    if tech_data.rsi is not None:
        total_indicators += 1
        if tech_data.rsi < 30:
            bullish_signals += 1
            reasons.append("RSI oversold")
        elif tech_data.rsi > 70:
            bearish_signals += 1
            reasons.append("RSI overbought")
        elif 45 <= tech_data.rsi <= 55:
            bullish_signals += 0.3
            bearish_signals += 0.3
    
    # EMA Analysis
    if tech_data.ema_8 and tech_data.ema_21:
        total_indicators += 1
        if tech_data.ema_8 > tech_data.ema_21:
            if current_price > tech_data.ema_8:
                bullish_signals += 1
                reasons.append("Precio sobre EMA alcista")
            else:
                bullish_signals += 0.5
                reasons.append("EMA alcista")
        else:
            if current_price < tech_data.ema_8:
                bearish_signals += 1
                reasons.append("Precio bajo EMA bajista")
            else:
                bearish_signals += 0.5
                reasons.append("EMA bajista")
    
    # Bollinger Bands
    if tech_data.bb_upper and tech_data.bb_lower and tech_data.bb_middle:
        total_indicators += 1
        if current_price <= tech_data.bb_lower:
            bullish_signals += 0.8
            reasons.append("BB oversold")
        elif current_price >= tech_data.bb_upper:
            bearish_signals += 0.8
            reasons.append("BB overbought")
        elif current_price > tech_data.bb_middle:
            bullish_signals += 0.3
        else:
            bearish_signals += 0.3
    
    # Volume confirmation
    if tech_data.volume_spike:
        if len(reasons) > 0:
            if bullish_signals > bearish_signals:
                bullish_signals += 0.5
                reasons.append("Volume alto")
            elif bearish_signals > bullish_signals:
                bearish_signals += 0.5
                reasons.append("Volume alto")
    
    if total_indicators == 0:
        return None, 0.0, "No hay indicadores válidos"
    
    bullish_ratio = bullish_signals / total_indicators
    bearish_ratio = bearish_signals / total_indicators
    
    signal_type = None
    base_confidence = 0.0
    min_threshold = 0.55
    
    if bullish_ratio > bearish_ratio and bullish_ratio >= min_threshold:
        if timeframe in [TimeFrame.M5, TimeFrame.M15]:
            signal_type = SignalType.SCALP_LONG
        elif timeframe in [TimeFrame.H4, TimeFrame.D1]:
            signal_type = SignalType.SWING_LONG
        else:
            signal_type = SignalType.LONG
        base_confidence = bullish_ratio * 100
    elif bearish_ratio > bullish_ratio and bearish_ratio >= min_threshold:
        if timeframe in [TimeFrame.M5, TimeFrame.M15]:
            signal_type = SignalType.SCALP_SHORT
        elif timeframe in [TimeFrame.H4, TimeFrame.D1]:
            signal_type = SignalType.SWING_SHORT
        else:
            signal_type = SignalType.SHORT
        base_confidence = bearish_ratio * 100
    
    if signal_type is None:
        return None, 0.0, "Señales insuficientes"
    
    tf_weight = TIMEFRAME_WEIGHTS.get(timeframe, 1.0)
    final_confidence = min(base_confidence * (1 + (tf_weight - 1) * 0.1), 95.0)
    
    reason_text = " + ".join(reasons[:3])
    
    return signal_type, final_confidence, reason_text

async def analyze_multi_timeframe_confluence(symbol: str) -> Optional[Signal]:
    """Análisis multi-timeframe con confluencia mejorada"""
    
    macro_context = {}
    timeframe_signals = {}
    
    for tf in active_timeframes:
        tech_data = await analyze_symbol_technical(symbol, tf)
        if tech_data:
            signal_type, confidence, reason = await calculate_signal_confidence(tech_data, symbol, tf, macro_context)
            if signal_type and confidence >= (min_confidence - 10):
                timeframe_signals[tf] = {
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'reason': reason,
                    'tech_data': tech_data
                }
    
    if not timeframe_signals:
        return None
    
    long_signals = []
    short_signals = []
    
    for tf, data in timeframe_signals.items():
        if data['signal_type'] in [SignalType.LONG, SignalType.SCALP_LONG, SignalType.SWING_LONG]:
            long_signals.append((tf, data))
        else:
            short_signals.append((tf, data))
    
    dominant_signals = long_signals if len(long_signals) >= len(short_signals) else short_signals
    
    if len(dominant_signals) < 1:
        return None
    
    if len(dominant_signals) == 1 and dominant_signals[0][1]['confidence'] < min_confidence:
        return None
    
    if len(dominant_signals) > 1:
        avg_confidence = sum(data['confidence'] for _, data in dominant_signals) / len(dominant_signals)
        if avg_confidence < (min_confidence - 5):
            return None
    
    total_weight = 0
    weighted_confidence = 0
    
    for tf, data in dominant_signals:
        weight = TIMEFRAME_WEIGHTS[tf]
        total_weight += weight
        weighted_confidence += data['confidence'] * weight
    
    final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
    
    if len(dominant_signals) > 1:
        confluence_bonus = min(len(dominant_signals) * 2, 8)
        final_confidence += confluence_bonus
        final_confidence = min(final_confidence, 95.0)
    
    if final_confidence < min_confidence:
        return None
    
    primary_tf = max(dominant_signals, key=lambda x: TIMEFRAME_WEIGHTS[x[0]])[0]
    primary_data = timeframe_signals[primary_tf]
    
    current_price = await get_current_price(symbol)
    if not current_price:
        return None
    
    if primary_tf in [TimeFrame.M5, TimeFrame.M15]:
        sl_pct = 0.8
        tp_pct = 1.6
    elif primary_tf == TimeFrame.H1:
        sl_pct = 1.2
        tp_pct = 2.4
    elif primary_tf == TimeFrame.H4:
        sl_pct = 2.0
        tp_pct = 4.0
    else:  # D1
        sl_pct = 3.0
        tp_pct = 6.0
    
    if primary_data['signal_type'] in [SignalType.LONG, SignalType.SCALP_LONG, SignalType.SWING_LONG]:
        stop_loss = current_price * (1 - sl_pct / 100)
        take_profit = current_price * (1 + tp_pct / 100)
    else:
        stop_loss = current_price * (1 + sl_pct / 100)
        take_profit = current_price * (1 - tp_pct / 100)
    
    risk_reward = tp_pct / sl_pct
    
    if len(dominant_signals) > 1:
        confluence_reason = f"Confluencia {len(dominant_signals)}TF: {primary_data['reason']}"
    else:
        confluence_reason = f"{primary_tf.value}: {primary_data['reason']}"
    
    signal = Signal(
        symbol=symbol,
        signal_type=primary_data['signal_type'],
        timeframe=primary_tf,
        confidence=final_confidence,
        entry_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=risk_reward,
        reason=confluence_reason,
        timestamp=datetime.utcnow(),
        indicators={
            f"{tf.value}_conf": data['confidence'] 
            for tf, data in dominant_signals
        }
    )
    
    return signal

# ============================================================================
# TELEGRAM BOT HANDLERS CON MANEJO ROBUSTO DE ERRORES
# ============================================================================

async def safe_send_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, **kwargs):
    """Envía mensaje con manejo robusto de errores y sin parse_mode problemático"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = await context.bot.send_message(chat_id=chat_id, text=text, **kwargs)
            return message
        except TimedOut:
            print(f"⚠️ Timeout enviando mensaje a {chat_id}, intento {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except RetryAfter as e:
            print(f"⚠️ Rate limit, esperando {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
        except NetworkError as e:
            print(f"⚠️ Error de red enviando a {chat_id}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Error enviando mensaje a {chat_id}: {e}")
            if attempt == max_retries - 1 and 'parse_mode' in kwargs:
                try:
                    kwargs_safe = {k: v for k, v in kwargs.items() if k != 'parse_mode'}
                    message = await context.bot.send_message(chat_id=chat_id, text=safe_markdown_text(text), **kwargs_safe)
                    return message
                except:
                    pass
            break
    
    return None

async def safe_edit_message(query, text: str, **kwargs) -> bool:
    """Edita mensaje con manejo robusto de errores"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await query.edit_message_text(text, **kwargs)
            return True
        except TimedOut:
            print(f"⚠️ Timeout editando mensaje, intento {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except RetryAfter as e:
            print(f"⚠️ Rate limit, esperando {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
        except NetworkError as e:
            print(f"⚠️ Error de red editando mensaje: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Error editando mensaje: {e}")
            if attempt == max_retries - 1 and 'parse_mode' in kwargs:
                try:
                    kwargs_safe = {k: v for k, v in kwargs.items() if k != 'parse_mode'}
                    await query.edit_message_text(safe_markdown_text(text), **kwargs_safe)
                    return True
                except:
                    pass
            break
    
    return False

def format_signal_message(signal: Signal) -> str:
    """Formatea una señal para envío por Telegram - VERSION SEGURA"""
    return create_safe_message(
        symbol=signal.symbol,
        signal_type=signal.signal_type.value,
        timeframe=signal.timeframe.value,
        confidence=signal.confidence,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        risk_reward=signal.risk_reward,
        reason=signal.reason
    )

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /start"""
    user = update.effective_user
    
    add_user(user.id, user.username, user.first_name)
    
    welcome_text = f"""🚀 ¡Bienvenido a GaryBot Pro! 🚀

Hola {user.first_name}! Soy tu asistente de trading avanzado con análisis multi-timeframe.

🎯 Características principales:
• Análisis técnico de indicadores múltiples
• Señales multi-timeframe con confluencia
• Sistema de confianza 0-100%
• Stop Loss y Take Profit optimizados
• Análisis automático cada 5 minutos

📊 Comandos disponibles:
/help - Ver todos los comandos
/status - Estado del bot
/symbols - Gestionar símbolos
/settings - Configuración

🔥 ¡El bot ya está escaneando los mejores setups para ti cada 5 minutos!"""
    
    keyboard = [
        [InlineKeyboardButton("📊 Ver Símbolos", callback_data="symbols")],
        [InlineKeyboardButton("⚙️ Configuración", callback_data="settings")],
        [InlineKeyboardButton("📈 Estado", callback_data="status")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await safe_send_message(context, update.message.chat.id, welcome_text, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /help"""
    help_text = """🤖 GaryBot Pro - Comandos Disponibles

📊 Trading:
/scan - Escanear mercado manualmente
/analyze [SYMBOL] - Analizar símbolo específico

⚙️ Configuración:
/symbols - Gestionar lista de símbolos
/settings - Configurar parámetros
/confidence [60-95] - Cambiar confianza mínima

📈 Información:
/status - Estado del bot y mercado
/top - Top símbolos por volumen

🛠️ Utilidades:
/start - Reiniciar bot
/help - Esta ayuda

Escaneo automático cada 5 minutos. ¿Necesitas ayuda específica? ¡Pregúntame!"""
    
    await safe_send_message(context, update.message.chat.id, help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /status"""
    user_id = update.effective_user.id
    update_user_activity(user_id)
    
    active_users = get_active_users()
    
    status_text = f"""🔥 Estado de GaryBot Pro

🤖 Bot:
• Estado: {'🟢 Activo' if running else '🔴 Inactivo'}
• Usuarios activos: {len(active_users)}
• Símbolos monitoreados: {len(selected_symbols)}
• Timeframes activos: {len(active_timeframes)}
• Confianza mínima: {min_confidence}%
• Intervalo escaneo: {SCAN_INTERVAL//60} minutos

⚙️ Configuración:
• Cache: {len(price_cache)} precios, {len(data_cache)} datasets
• Timeframes: {', '.join([tf.value for tf in active_timeframes])}

💾 Base de datos:
• Usuarios: {len(users_cache)}
• Señales en historial: Disponible"""
    
    keyboard = [
        [InlineKeyboardButton("🔄 Actualizar", callback_data="status")],
        [InlineKeyboardButton("📊 Símbolos", callback_data="symbols")],
        [InlineKeyboardButton("⚙️ Config", callback_data="settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await safe_edit_message(update.callback_query, status_text, reply_markup=reply_markup)
    else:
        await safe_send_message(context, update.message.chat.id, status_text, reply_markup=reply_markup)

async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para gestión de símbolos"""
    user_id = update.effective_user.id
    update_user_activity(user_id)
    
    if len(selected_symbols) == 0:
        top_symbols = get_top_symbols_by_volume(20)
        selected_symbols.extend(top_symbols[:10])
        save_preferences()
    
    symbols_text = f"""📊 Símbolos Monitoreados ({len(selected_symbols)})

Activos:
{', '.join(selected_symbols[:15])}
{f"... y {len(selected_symbols) - 15} más" if len(selected_symbols) > 15 else ""}

💡 Opciones:
• Añadir símbolo: Escribe el símbolo (ej: ADAUSDT)
• Cargar top por volumen: Usa los botones
• Limpiar lista: Usa botón correspondiente"""
    
    keyboard = [
        [InlineKeyboardButton("🔍 Cargar Top 20", callback_data="load_top_20")],
        [InlineKeyboardButton("🧹 Limpiar Lista", callback_data="clear_symbols")],
        [InlineKeyboardButton("📈 Escanear Todos", callback_data="scan_all")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await safe_edit_message(update.callback_query, symbols_text, reply_markup=reply_markup)
    else:
        await safe_send_message(context, update.message.chat.id, symbols_text, reply_markup=reply_markup)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para configuración"""
    user_id = update.effective_user.id
    update_user_activity(user_id)
    
    settings_text = f"""⚙️ Configuración de GaryBot Pro

📊 Parámetros actuales:
• Confianza mínima: {min_confidence}%
• Timeframes activos: {len(active_timeframes)}/5
• Intervalo escaneo: {SCAN_INTERVAL//60} minutos

🕒 Timeframes disponibles:
{' '.join(['✅' if tf in active_timeframes else '❌' for tf in TimeFrame]) + ' ' + ' '.join([tf.value for tf in TimeFrame])}

💡 Comandos de configuración:
• /confidence [60-95] - Cambiar confianza mínima
• Usar botones para cambiar configuración"""
    
    keyboard = [
        [InlineKeyboardButton("📊 Confianza", callback_data="set_confidence")],
        [InlineKeyboardButton("🕒 Timeframes", callback_data="set_timeframes")],
        [InlineKeyboardButton("🔙 Volver", callback_data="status")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await safe_edit_message(update.callback_query, settings_text, reply_markup=reply_markup)
    else:
        await safe_send_message(context, update.message.chat.id, settings_text, reply_markup=reply_markup)

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /analyze [SYMBOL] con manejo robusto"""
    if not context.args:
        help_text = (
            "📊 Uso: /analyze BTCUSDT\n\n"
            "Ejemplos:\n"
            "• /analyze ETHUSDT\n"
            "• /analyze ADAUSDT\n"
            "• /analyze SOLUSDT"
        )
        await safe_send_message(context, update.message.chat.id, help_text)
        return
    
    symbol = context.args[0].upper().strip()
    
    if not is_valid_trading_symbol(symbol):
        error_text = (
            "❌ Símbolo inválido\n\n"
            "Usa formato como: BTCUSDT, ETHUSDT, ADAUSDT, etc."
        )
        await safe_send_message(context, update.message.chat.id, error_text)
        return
    
    update_user_activity(update.effective_user.id)
    
    loading_text = f"🔍 Analizando {symbol}...\n\nObteniendo datos de {len(active_timeframes)} timeframes..."
    loading_msg_obj = await safe_send_message(context, update.message.chat.id, loading_text)
    
    try:
        signal = await analyze_multi_timeframe_confluence(symbol)
        
        if signal:
            result_text = f"🎯 Análisis Completo de {symbol}\n\n" + format_signal_message(signal)
            
            result_text += f"\n\n📊 Desglose por Timeframes:\n"
            for tf in active_timeframes:
                tech_data = await analyze_symbol_technical(symbol, tf)
                if tech_data:
                    macro_context = {}
                    signal_type, confidence, reason = await calculate_signal_confidence(tech_data, symbol, tf, macro_context)
                    if confidence > 0:
                        status_icon = "✅" if confidence >= min_confidence else "⚠️"
                        result_text += f"{status_icon} {tf.value}: {confidence:.1f}% - {reason}\n"
            
        else:
            result_text = f"📊 Análisis Detallado de {symbol}\n\n"
            
            current_price = await get_current_price(symbol)
            if current_price:
                result_text += f"💰 Precio actual: ${current_price:.6f}\n\n"
            
            result_text += f"Análisis por timeframes:\n"
            
            any_signal = False
            for tf in active_timeframes:
                tech_data = await analyze_symbol_technical(symbol, tf)
                if tech_data:
                    macro_context = {}
                    signal_type, confidence, reason = await calculate_signal_confidence(tech_data, symbol, tf, macro_context)
                    
                    if confidence > 0:
                        status_icon = "✅" if confidence >= min_confidence else "⚠️" 
                        result_text += f"{status_icon} {tf.value}: {confidence:.1f}% - {reason}\n"
                        if confidence >= min_confidence:
                            any_signal = True
                    else:
                        result_text += f"❌ {tf.value}: Sin señal clara\n"
                else:
                    result_text += f"⚠️ {tf.value}: Error obteniendo datos\n"
            
            if not any_signal:
                result_text += f"\n💡 Ningún timeframe alcanza {min_confidence}% de confianza\n"
                result_text += f"Considera reducir el threshold con /confidence [valor]"
        
        await safe_send_message(context, update.message.chat.id, result_text)
        
    except Exception as e:
        error_text = f"❌ Error analizando {symbol}\n\n{str(e)}"
        await safe_send_message(context, update.message.chat.id, error_text)

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /scan con manejo robusto"""
    await manual_scan_command(update, context)

async def confidence_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /confidence [valor] con manejo robusto"""
    global min_confidence
    
    if not context.args:
        current_text = (
            f"🎯 Confianza mínima actual: {min_confidence}%\n\n"
            f"Para cambiar: /confidence [60-95]\n\n"
            f"Ejemplos:\n"
            f"• /confidence 70 (recomendado)\n"
            f"• /confidence 80 (conservador)\n"
            f"• /confidence 65 (más señales)"
        )
        await safe_send_message(context, update.message.chat.id, current_text)
        return
    
    try:
        new_confidence = int(context.args[0])
        if 60 <= new_confidence <= 95:
            old_confidence = min_confidence
            min_confidence = new_confidence
            save_preferences()
            
            success_text = (
                f"✅ Confianza actualizada\n\n"
                f"Anterior: {old_confidence}%\n"
                f"Nueva: {min_confidence}%\n\n"
                f"Este cambio afecta a:\n"
                f"• Señales automáticas\n"
                f"• Escaneos manuales\n"
                f"• Comando /analyze"
            )
            await safe_send_message(context, update.message.chat.id, success_text)
        else:
            error_text = (
                "❌ Valor fuera de rango\n\n"
                "La confianza debe estar entre 60% y 95%"
            )
            await safe_send_message(context, update.message.chat.id, error_text)
    except ValueError:
        error_text = (
            "❌ Valor inválido\n\n"
            "Usa un número entero entre 60 y 95.\n"
            "Ejemplo: /confidence 75"
        )
        await safe_send_message(context, update.message.chat.id, error_text)
    
    update_user_activity(update.effective_user.id)

async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para comando /top con manejo robusto"""
    update_user_activity(update.effective_user.id)
    
    loading_text = "📊 Obteniendo top símbolos...\n\nConsultando Binance API..."
    loading_msg_obj = await safe_send_message(context, update.message.chat.id, loading_text)
    
    try:
        top_symbols = get_top_symbols_by_volume(20)
        
        top_text = "🔍 Top 20 Símbolos por Volumen (24h)\n\n"
        
        for i, symbol in enumerate(top_symbols, 1):
            price = await get_current_price(symbol)
            price_str = f"${price:.4f}" if price else "N/A"
            
            if i == 1:
                emoji = "🥇"
            elif i == 2:
                emoji = "🥈"
            elif i == 3:
                emoji = "🥉"
            else:
                emoji = f"{i:2d}."
            
            top_text += f"{emoji} {symbol} - {price_str}\n"
        
        top_text += f"\n💡 Usa los botones para cargar estos símbolos"
        
        keyboard = [
            [InlineKeyboardButton("🔥 Cargar Top 10", callback_data="load_top_10"),
             InlineKeyboardButton("🔥 Cargar Top 20", callback_data="load_top_20")],
            [InlineKeyboardButton("📊 Ver Mis Símbolos", callback_data="symbols")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await safe_send_message(context, update.message.chat.id, top_text, reply_markup=reply_markup)
        
    except Exception as e:
        error_text = f"❌ Error obteniendo símbolos\n\n{str(e)}"
        await safe_send_message(context, update.message.chat.id, error_text)

async def manual_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Escaneo manual desde comando con manejo robusto"""
    user_id = update.effective_user.id
    update_user_activity(user_id)
    
    if not selected_symbols:
        error_text = (
            "❌ No hay símbolos configurados\n\n"
            "Usa /symbols para añadir símbolos o /top para cargar los más populares."
        )
        await safe_send_message(context, update.message.chat.id, error_text)
        return
    
    initial_text = (
        f"🔍 Iniciando escaneo...\n\n"
        f"Símbolos: {len(selected_symbols)}\n"
        f"Timeframes: {len(active_timeframes)}\n"
        f"Confianza mínima: {min_confidence}%\n\n"
        f"Esto puede tomar 15-45 segundos..."
    )
    loading_msg_obj = await safe_send_message(context, update.message.chat.id, initial_text)
    
    signals_found = []
    symbols_processed = 0
    
    batch_size = 5
    total_symbols = min(len(selected_symbols), 20)
    
    for i in range(0, total_symbols, batch_size):
        batch = selected_symbols[i:i + batch_size]
        
        for symbol in batch:
            try:
                signal = await analyze_multi_timeframe_confluence(symbol)
                if signal:
                    signals_found.append(signal)
            except Exception as e:
                print(f"Error escaneando {symbol}: {e}")
            
            symbols_processed += 1
        
        progress = (symbols_processed / total_symbols) * 100
        progress_text = (
            f"🔍 Progreso: {progress:.0f}%\n\n"
            f"Procesados: {symbols_processed}/{total_symbols}\n"
            f"Señales encontradas: {len(signals_found)}\n\n"
            f"Analizando {', '.join(batch)}..."
        )
        
        if loading_msg_obj:
            try:
                await context.bot.edit_message_text(
                    chat_id=update.message.chat.id,
                    message_id=loading_msg_obj.message_id,
                    text=progress_text
                )
            except:
                pass
    
    if signals_found:
        signals_found.sort(key=lambda x: x.confidence, reverse=True)
        
        result_text = f"🎯 Escaneo Completado - {len(signals_found)} Señales\n\n"
        
        for i, signal in enumerate(signals_found[:5], 1):
            result_text += f"{i}. {signal.symbol} - {signal.confidence:.1f}%\n"
            result_text += f"📊 {signal.signal_type.value} ({signal.timeframe.value})\n"
            result_text += f"💰 ${signal.entry_price:.6f} → ${signal.take_profit:.6f}\n"
            result_text += f"🛑 SL: ${signal.stop_loss:.6f} | R:R 1:{signal.risk_reward:.1f}\n\n"
        
        if len(signals_found) > 5:
            result_text += f"➕ {len(signals_found) - 5} señales adicionales\n"
        
        result_text += f"\n📈 Stats: {symbols_processed} símbolos escaneados"
        
    else:
        result_text = f"📊 Escaneo Completado - Sin Señales\n\n"
        result_text += f"Símbolos analizados: {symbols_processed}\n\n"
        result_text += f"Criterios actuales:\n"
        result_text += f"• Confianza mínima: {min_confidence}%\n"
        result_text += f"• Timeframes: {', '.join([tf.value for tf in active_timeframes])}\n\n"
        result_text += f"Sugerencias:\n"
        result_text += f"• Reducir confianza: /confidence 65\n"
        result_text += f"• Añadir más símbolos: /top\n"
        result_text += f"• Intentar más tarde cuando haya más movimiento"
    
    await safe_send_message(context, update.message.chat.id, result_text)

# ============================================================================
# CALLBACK QUERY HANDLERS
# ============================================================================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para botones inline"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    update_user_activity(user_id)
    
    data = query.data
    
    try:
        if data == "status":
            await status_command(update, context)
        
        elif data == "symbols":
            await symbols_command(update, context)
        
        elif data == "settings":
            await settings_command(update, context)
        
        elif data == "load_top_10":
            top_symbols = get_top_symbols_by_volume(20)
            selected_symbols.clear()
            selected_symbols.extend(top_symbols[:10])
            save_preferences()
            
            success_text = f"✅ Cargados Top 10 símbolos\n\n{', '.join(selected_symbols)}"
            await safe_edit_message(query, success_text)
        
        elif data == "load_top_20":
            top_symbols = get_top_symbols_by_volume(20)
            selected_symbols.clear()
            selected_symbols.extend(top_symbols)
            save_preferences()
            
            success_text = f"✅ Cargados Top 20 símbolos\n\n{', '.join(selected_symbols[:15])}..."
            await safe_edit_message(query, success_text)
        
        elif data == "clear_symbols":
            selected_symbols.clear()
            save_preferences()
            
            clear_text = "🧹 Lista de símbolos limpia\n\nUsa /top para cargar símbolos populares"
            await safe_edit_message(query, clear_text)
        
        elif data == "scan_all":
            await manual_scan_command(update, context)
        
        elif data == "set_confidence":
            conf_text = f"""📊 Configurar Confianza Mínima

Actual: {min_confidence}%

Niveles recomendados:
• 65% - Más señales, más ruido
• 70% - Equilibrado (recomendado)
• 75% - Conservador
• 80% - Muy selectivo

Para cambiar: /confidence [valor]"""
            await safe_edit_message(query, conf_text)
        
        elif data == "set_timeframes":
            tf_text = f"""🕒 Configurar Timeframes

Activos: {', '.join([tf.value for tf in active_timeframes])}

Disponibles:
• 5m - Scalping rápido
• 15m - Scalping/intraday
• 1h - Trading intraday
• 4h - Swing trading
• 1d - Trading posicional

💡 Usa comandos para activar/desactivar timeframes específicos"""
            await safe_edit_message(query, tf_text)
        
    except Exception as e:
        error_text = f"❌ Error procesando acción: {str(e)}"
        await safe_edit_message(query, error_text)

# ============================================================================
# MESSAGE HANDLERS PARA AÑADIR SÍMBOLOS
# ============================================================================

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para mensajes de texto - añadir símbolos"""
    if not update.message or not update.message.text:
        return
    
    user_id = update.effective_user.id
    update_user_activity(user_id)
    
    text = update.message.text.upper().strip()
    
    if is_valid_trading_symbol(text) and text not in selected_symbols:
        price = await get_current_price(text)
        if price:
            selected_symbols.append(text)
            save_preferences()
            
            success_text = f"✅ {text} añadido\n\nPrecio actual: ${price:.6f}"
            await safe_send_message(context, update.message.chat.id, success_text)
        else:
            error_text = f"❌ {text} no encontrado\n\nVerifica que el símbolo existe en Binance"
            await safe_send_message(context, update.message.chat.id, error_text)
    
    elif text in selected_symbols:
        info_text = f"ℹ️ {text} ya está en la lista\n\nUsa /symbols para ver todos"
        await safe_send_message(context, update.message.chat.id, info_text)

# ============================================================================
# SISTEMA DE ESCANEO AUTOMÁTICO MEJORADO (5 MINUTOS)
# ============================================================================

async def automatic_scan_loop():
    """Loop principal de escaneo automático - CADA 5 MINUTOS"""
    global running
    
    print(f"🔄 Iniciando sistema de escaneo automático cada {SCAN_INTERVAL//60} minutos...")
    
    while running:
        try:
            if not selected_symbols:
                print("⚠️ No hay símbolos configurados para escaneo automático")
                await asyncio.sleep(300)
                continue
            
            print(f"🔍 Iniciando escaneo automático de {len(selected_symbols)} símbolos...")
            scan_start_time = datetime.now()
            
            active_users = get_active_users()
            if not active_users:
                print("⚠️ No hay usuarios activos")
                await asyncio.sleep(SCAN_INTERVAL)
                continue
            
            signals_found = []
            
            batch_size = 3
            scanned_count = 0
            
            for i in range(0, min(len(selected_symbols), 25), batch_size):
                if not running:
                    break
                
                batch = selected_symbols[i:i + batch_size]
                
                for symbol in batch:
                    try:
                        signal = await analyze_multi_timeframe_confluence(symbol)
                        if signal:
                            signals_found.append(signal)
                            print(f"🎯 Señal encontrada: {signal.symbol} - {signal.confidence:.1f}%")
                        scanned_count += 1
                    except Exception as e:
                        print(f"⚠️ Error escaneando {symbol}: {e}")
                        scanned_count += 1
                
                await asyncio.sleep(1)
            
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            
            if signals_found:
                signals_found.sort(key=lambda x: x.confidence, reverse=True)
                
                print(f"📤 Enviando {len(signals_found)} señales a {len(active_users)} usuarios...")
                
                for signal in signals_found[:3]:
                    message = create_safe_message(
                        symbol=signal.symbol,
                        signal_type=signal.signal_type.value,
                        timeframe=signal.timeframe.value,
                        confidence=signal.confidence,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        risk_reward=signal.risk_reward,
                        reason=signal.reason
                    )
                    
                    sent_count = 0
                    failed_count = 0
                    
                    for user in active_users:
                        try:
                            if app:
                                await app.bot.send_message(
                                    chat_id=user.user_id,
                                    text=message,
                                )
                                sent_count += 1
                        except Exception as e:
                            print(f"⚠️ Error enviando a usuario {user.user_id}: {e}")
                            failed_count += 1
                    
                    save_signal_to_history(signal, sent_count)
                    
                    print(f"📤 Señal {signal.symbol} enviada a {sent_count} usuarios ({failed_count} fallos)")
                    
                    await asyncio.sleep(3)
            
            else:
                print(f"📊 Escaneo completado en {scan_duration:.1f}s - {scanned_count} símbolos - No se encontraron señales")
            
            next_scan_time = datetime.now() + timedelta(seconds=SCAN_INTERVAL)
            print(f"⏰ Próximo escaneo a las {next_scan_time.strftime('%H:%M:%S')}")
            
            for _ in range(SCAN_INTERVAL):
                if not running:
                    break
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"❌ Error en escaneo automático: {e}")
            await asyncio.sleep(60)

# ============================================================================
# FUNCIÓN PRINCIPAL Y CONFIGURACIÓN DEL BOT
# ============================================================================

async def main():
    """Función principal del bot"""
    global app, running, TELEGRAM_BOT_TOKEN, BINANCE_API_KEY, BINANCE_API_SECRET
    
    print("🚀 Iniciando GaryBot Pro v4.1...")
    print(f"⏰ Configurado para escaneo cada {SCAN_INTERVAL//60} minutos")
    
    # Cargar variables de entorno
    load_dotenv()
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN no configurado en .env")
        return
    
    print("🔗 Probando conexión...")
    connected, message = await test_connection()
    if not connected:
        print(f"❌ {message}")
        return
    else:
        print(f"✅ {message}")
    
    print("🗄️ Inicializando base de datos...")
    init_database()
    
    print("⚙️ Cargando preferencias...")
    load_preferences()
    
    if not selected_symbols:
        print("📊 Cargando símbolos por defecto...")
        top_symbols = get_top_symbols_by_volume(15)
        selected_symbols.extend(top_symbols[:10])
        save_preferences()
        print(f"✅ Cargados {len(selected_symbols)} símbolos")
    
    print("🤖 Configurando aplicación Telegram...")
    request = create_telegram_request()
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).request(request).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("symbols", symbols_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("confidence", confidence_command))
    app.add_handler(CommandHandler("top", top_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    print("✅ Handlers configurados")
    
    running = True
    scan_task = None
    
    try:
        print("🔄 Iniciando polling...")
        
        await app.initialize()
        await app.start()
        await app.updater.start_polling(
            allowed_updates=['message', 'callback_query'],
            drop_pending_updates=True
        )
        
        scan_task = asyncio.create_task(automatic_scan_loop())
        
        print("🟢 GaryBot Pro está funcionando!")
        print(f"📊 Escaneo automático cada {SCAN_INTERVAL//60} minutos activado")
        print("🛑 Presiona Ctrl+C para detener")
        
        while running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo bot...")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        running = False
        
        if scan_task and not scan_task.done():
            scan_task.cancel()
            try:
                await scan_task
            except asyncio.CancelledError:
                print("✅ Escaneo automático cancelado")
        
        try:
            if app and hasattr(app, 'updater') and app.updater.running:
                await app.updater.stop()
                await app.stop()
            if app:
                await app.shutdown()
        except Exception as e:
            print(f"⚠️ Error deteniendo bot: {e}")
        
        print("✅ Bot detenido correctamente")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando bot: {e}")
        print(f"Detalles: {str(e)}")
        import traceback
        traceback.print_exc()