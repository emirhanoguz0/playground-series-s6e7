# S6E7 Baseline: XGBoost + native NaN + prior correction
# CV: StratifiedKFold(5), metrik: balanced accuracy (OOF)
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

DATA = "/sessions/gifted-wizardly-sagan/mnt/playground-series-s6e7"
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

target_map = {"at-risk": 0, "unhealthy": 1, "fit": 2}
inv_map = {v: k for k, v in target_map.items()}
y = train["health_condition"].map(target_map).values

feats = [c for c in train.columns if c not in ("id", "health_condition")]
X = train[feats].copy()
X_test = test[feats].copy()

# kategorikler: category dtype (NaN korunur, doldurma YOK)
for c in feats:
    if not is_numeric_dtype(X[c]):
        X[c] = X[c].astype("category")
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)

params = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=-1,
)

priors = np.bincount(y) / len(y)  # prior correction icin
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros((len(X), 3))
test_proba = np.zeros((len(X_test), 3))

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = XGBClassifier(**params)
    model.fit(
        X.iloc[tr_idx], y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        verbose=False,
    )
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])
    test_proba += model.predict_proba(X_test) / 5
    # fold skoru (prior correction ile)
    pred = np.argmax(oof[va_idx] / priors, axis=1)
    ba = balanced_accuracy_score(y[va_idx], pred)
    print(f"fold {fold}: BA={ba:.5f}  best_iter={model.best_iteration}  ({time.time()-t0:.0f}s)")

# OOF skorlari: duz argmax vs prior correction
ba_plain = balanced_accuracy_score(y, np.argmax(oof, axis=1))
ba_prior = balanced_accuracy_score(y, np.argmax(oof / priors, axis=1))
print(f"\nOOF BA (duz argmax):        {ba_plain:.5f}")
print(f"OOF BA (prior correction):  {ba_prior:.5f}")

# submission
sub_pred = np.argmax(test_proba / priors, axis=1)
sub = pd.DataFrame({"id": test["id"], "health_condition": [inv_map[p] for p in sub_pred]})
sub.to_csv("/sessions/gifted-wizardly-sagan/work/submission_baseline.csv", index=False)
print("\nsubmission dagilimi:")
print(sub.health_condition.value_counts(normalize=True).round(4))
np.save("/sessions/gifted-wizardly-sagan/work/oof_baseline.npy", oof)
np.save("/sessions/gifted-wizardly-sagan/work/test_proba_baseline.npy", test_proba)
print(f"\ntoplam sure: {time.time()-t0:.0f}s")