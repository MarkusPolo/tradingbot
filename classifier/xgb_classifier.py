import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import os


def load_datasets():
    # Features und Regime laden
    df_feat = pd.read_csv('../data/features/BTC_features.csv', parse_dates=['timestamp'])
    df_reg = pd.read_csv('../data/regime/BTC_regimes.csv', parse_dates=['timestamp'])
    # Nur relevante Spalten: timestamp und regime
    df_reg = df_reg[['timestamp', 'regime']]
    # Zusammenführen anhand timestamp
    df = pd.merge(df_feat, df_reg, on='timestamp', how='inner')
    # Zielvariable: Rendite in der nächsten Periode
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df


def train_xgb(df):
    # Feature-Set: alle außer timestamp, close, target
    X = df.drop(columns=['timestamp', 'close', 'target'])
    y = df['target']
    # Zeitlich-sequentielles Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # Modell initialisieren
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    # Training
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Speichern
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/xgb_classifier.pkl')
    print('XGBoost-Modell gespeichert unter ../models/xgb_classifier.pkl')

if __name__ == '__main__':
    df = load_datasets()
    train_xgb(df)
