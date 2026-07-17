# Deney 8: stacking
# XGB(exp3) + LGBM + CatBoost OOF olasiliklari uzerine meta-model
# (multinomial lojistik regresyon). Ayni CV bolmesi (seed=42):
# meta-model her fold'da diger fold'larin OOF'uyla egitilir -> sizinti yok.
# Test tarafi: meta-model tum OOF ile egitilip test olasiliklarina uygulanir.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

DATA = "/sessions/gifted-wizardly-sagan/mnt/playground-series-s6e7"
WORK = "/sessions/gifted-wizardly-sagan/work"
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv", usecols=["health_condition"])
test = pd.read_csv(f"{DATA}/test.csv", usecols=["id"])
target_map = {"at-risk": 0, "unhealthy": 1, "fit": 2}
inv_map = {v: k for k, v in target_map.items()}
y = train["health_condition"].map(target_map).values
priors = np.bincount(y) / len(y)

# taban model ciktilari
Z = np.hstack([
    np.load(f"{WORK}/oof_features.npy"),   # XGB exp3
    np.load(f"{WORK}/oof_lgbm.npy"),
    np.load(f"{WORK}/oof_catboost.npy"),
])
Z_test = np.hstack([
    np.load(f"{WORK}/test_proba_features.npy"),
    np.load(f"{WORK}/test_proba_lgbm.npy"),
    np.load(f"{WORK}/test_proba_catboost.npy"),
])
# log-odds daha iyi calisir
eps = 1e-7
Zl = np.log(np.clip(Z, eps, 1))
Zl_test = np.log(np.clip(Z_test, eps, 1))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
meta_oof = np.zeros((len(y), 3))
for fold, (tr, va) in enumerate(skf.split(Zl, y)):
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Zl[tr], y[tr])
    meta_oof[va] = lr.predict_proba(Zl[va])
    ba = balanced_accuracy_score(y[va], np.argmax(meta_oof[va] / priors, axis=1))
    print(f"fold {fold}: BA={ba:.5f}", flush=True)

ba = balanced_accuracy_score(y, np.argmax(meta_oof / priors, axis=1))
print(f"\nstack OOF BA: {ba:.5f}")
print(f"exp3: 0.94988 | exp4 blend: 0.94993")

lr_full = LogisticRegression(max_iter=2000, C=1.0)
lr_full.fit(Zl, y)
test_proba = lr_full.predict_proba(Zl_test)
sub_pred = np.argmax(test_proba / priors, axis=1)
sub = pd.DataFrame({"id": test["id"], "health_condition": [inv_map[p] for p in sub_pred]})
sub.to_csv(f"{WORK}/submission_stack.csv", index=False)
np.save(f"{WORK}/test_proba_stack.npy", test_proba)

old = pd.read_csv(f"{DATA}/2-submission_features.csv")
diff = (old["health_condition"].values != sub["health_condition"].values).mean()
print(f"exp3 submission'dan fark: {diff:.4%}")
