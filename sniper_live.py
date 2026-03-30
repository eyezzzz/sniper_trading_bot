import ccxt
import os
from dotenv import load_dotenv
import pandas as pd
import joblib
import time
import yfinance as yf

load_dotenv()

# Config Binance Testnet
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
    'secret': os.getenv('BINANCE_TESTNET_SECRET_KEY'),
    'sandbox': True,  # TESTNET
    'enableRateLimit': True,
})

# Carrega modelo
modelo = joblib.load('sniper_cerebro.pkl')

def get_features():
    """Pega última vela + calcula features"""
    df = yf.download("BTC-USD", period="60d", interval="1h")
    df['Media_Curta'] = df['Close'].rolling(20).mean()
    df['Media_Longa'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    
    ultima = df.iloc[-1][['RSI', 'Media_Curta', 'Media_Longa', 'Volume']].values.reshape(1,-1)
    return ultima

def main():
    print("🤖 SNIPER BOT LIVE - TESTNET")
    print("Saldo atual:", exchange.fetch_balance())
    
    while True:
        try:
            features = get_features()
            sinal = modelo.predict(features)[0]
            preco = exchange.fetch_ticker('BTC/USDT')['last']
            
            print(f"💹 BTC: ${preco:.2f} | Sinal: {'🟢 COMPRA' if sinal==1 else '🔴 AGUARDA'}")
            
            if sinal == 1:
                print("🚀 SINAL DE COMPRA! (Testnet)")
                exchange.create_market_buy_order('BTC/USDT', 0.001)  # DESCOMENTE PARA TRADES REAIS
                
            time.sleep(10)  # 5 min
        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()