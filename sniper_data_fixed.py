import yfinance as yf
import pandas as pd

def baixar_dados_estrategicos():
    print("Baixando BTC-USD 1H...")
    df = yf.download("BTC-USD", period="60d", interval="1h")
    
    df['Media_Curta'] = df['Close'].rolling(20).mean()
    df['Media_Longa'] = df['Close'].rolling(50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    df.to_csv("dados_sniper.csv")
    print(f"✅ {len(df)} linhas salvas em dados_sniper.csv")
    print(df.tail())
    return df

if __name__ == "__main__":
    baixar_dados_estrategicos()