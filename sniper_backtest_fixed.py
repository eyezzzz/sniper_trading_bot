import pandas as pd
import joblib
import numpy as np

def backtest_sniper():
    print("🔍 Executando BACKTEST...")
    
    # Carrega e limpa dados
    df = pd.read_csv("dados_sniper.csv")
    
    # Força colunas numéricas
    cols_num = ['Close', 'RSI', 'Media_Curta', 'Media_Longa', 'Volume']
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)
    modelo = joblib.load("sniper_cerebro.pkl")
    
    features = ['RSI', 'Media_Curta', 'Media_Longa', 'Volume']
    X = df[features]
    
    # Previsões
    df['Previsao'] = modelo.predict(X)
    
    # Simula trades
    capital = 100.0
    taxa = 0.001
    trades = 0
    lucros = []
    
    for i in range(len(df)-1):
        if df['Previsao'].iloc[i] == 1:
            preco_atual = df['Close'].iloc[i]
            preco_futuro = df['Close'].iloc[i+1]
            variacao = (preco_futuro - preco_atual) / preco_atual
            capital *= (1 + variacao - (taxa * 2))
            trades += 1
            lucros.append(variacao)
    
    lucro_total = capital - 100.0
    win_rate = len([l for l in lucros if l > 0]) / len(lucros) if lucros else 0
    
    print(f"\n📊 RESULTADO DO BACKTEST:")
    print(f"Capital inicial: R$ 100,00")
    print(f"Capital final: R$ {capital:.2f}")
    print(f"Lucro/Prejuízo: R$ {lucro_total:+.2f} ({lucro_total/100*100:+.1f}%)")
    print(f"Total trades: {trades}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Retorno médio por trade: {np.mean(lucros)*100:.2f}%" if lucros else "Sem trades")

if __name__ == "__main__":
    backtest_sniper()