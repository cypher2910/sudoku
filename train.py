import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_AVAILABLE = False
from joblib import dump
from sudoku import generate_dataset

def main():
    os.makedirs("models", exist_ok=True)
    X, y = generate_dataset(n=500)
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    dump(rf, os.path.join("models", "random_forest.pkl"))
    if XGB_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=729,
            tree_method="hist",
            random_state=42,
        )
        xgb.fit(X, y)
        dump(xgb, os.path.join("models", "xgboost.pkl"))
    else:
        hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1)
        hgb.fit(X, y)
        dump(hgb, os.path.join("models", "xgboost.pkl"))

if __name__ == "__main__":
    main()
