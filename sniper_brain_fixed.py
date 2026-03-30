import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def treinar_sniper():
    print("Treinando o Sniper Brain...")
    
    # Carrega dados
    df = pd.read_csv("dados_sniper.csv", index_col=0, parse_dates=True)
    
    # Cria target: 1 se próxima close > atual
    df['Alvo'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    # Features
    features = ['RSI', 'Media_Curta', 'Media_Longa', 'Volume']
    X = df[features]
    y = df['Alvo']
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treina modelo
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Avalia
    previsoes = modelo.predict(X_test)
    precisao = accuracy_score(y_test, previsoes)
    
    print(f"✅ Precisão: {precisao*100:.2f}%")
    print(f"📊 Features usadas: {features}")
    
    # Salva modelo
    joblib.dump(modelo, "sniper_cerebro.pkl")
    print("🧠 Modelo salvo em sniper_cerebro.pkl")
    
    return modelo, precisao

if __name__ == "__main__":
    treinar_sniper()