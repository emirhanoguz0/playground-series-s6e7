# Deney 6: seed ortalamasi (kaldigi yerden devam edebilir)
# exp3 modeli 5 farkli seed ile egitilir, test olasiliklari ortalanir.
# Her seed'in sonucu ayri .npy'a kaydedilir; dosya varsa o seed atlanir.
# seed 42 = exp3'un birebir aynisi, hazir dosyadan alinir.
import os
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

DATA = "/sessions/gifted-wizardly-sagan/mnt/playground-series-s6e7"
WORK = "/sessions/gifted-wizardly-sagan/work"
SEEDS = [42, 7, 123, 2024, 555]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
target_map = {"at-risk": 0, "unhealthy": 1, "fit": 2}
inv_map = {v: k for k, v in target_map.items()}
y = train["health_condition"].map(target_map).values

def add_features(df):
    out = df.copy()
    s = out["sleep_duration"]
    out["sleep_lt6"] = (s < 6).astype("float")
    out["sleep_lt7"] = (s < 7).astype("float")
    out.loc[s.isna(), ["sleep_lt6", "sleep_lt7"]] = np.nan
    stress = out["stress_level"]
    act = out["physical_activity_level"]
    r_unh = ((s < 6) & (stress == "high")).astype("float")
    r_unh[s.isna() | stress.isna()] = np.nan
    out["rule_unhealthy"] = r_unh
    r_fit = ((s >= 7) & (stress == "low") & (act == "active")).astype("float")
    r_fit[s.isna() | stress.isna() | act.isna()] = np.nan
    out["rule_fit"] = r_fit
    band = pd.cut(s, [-np.inf, 6, 7, np.inf], labels=["lt6", "6to7", "ge7"]).astype("object")
    out["stress_sleep"] = stress.astype("object").fillna("na") + "_" + pd.Series(band, index=out.index).fillna("na")
    return out

ftr, fte = add_features(train), add_features(test)
feat_cols = [c for c in ftr.columns if c not in ("id", "health_condition")]
X, X_test = ftr[feat_cols].copy(), fte[feat_cols].copy()
for c in feat_cols:
    if not is_numeric_dtype(X[c]):
        X[c] = X[c].astype("category")
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)

priors = np.bincount(y) / len(y)
t0 = time.time()

# seed 42: exp3 ciktilari birebir ayni kosum
if not os.path.exists(f"{WORK}/tp_seed42.npy"):
    oof42 = np.load(f"{WORK}/oof_features.npy")
    ba42 = balanced_accuracy_score(y, np.argmax(oof42 / priors, axis=1))
    np.save(f"{WORK}/tp_seed42.npy", np.load(f"{WORK}/test_proba_features.npy"))
    np.save(f"{WORK}/ba_seed42.npy", np.array([ba42]))
    print(f"seed 42: exp3'ten alindi, OOF BA={ba42:.5f}", flush=True)

for seed in SEEDS:
    if os.path.exists(f"{WORK}/tp_seed{seed}.npy"):
        ba = float(np.load(f"{WORK}/ba_seed{seed}.npy")[0])
        print(f"seed {seed}: hazir (OOF BA={ba:.5f})", flush=True)
        continue
    params = dict(
        n_estimators=2000, learning_rate=0.05, max_depth=6, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
        enable_categorical=True, early_stopping_rounds=100, random_state=seed, n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), 3))
    tp = np.zeros((len(X_test), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        m = XGBClassifier(**params)
        m.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])
        tp += m.predict_proba(X_test) / 5
        print(f"  seed {seed} fold {fold} tamam ({time.time()-t0:.0f}s)", flush=True)
    ba = balanced_accuracy_score(y, np.argmax(oof / priors, axis=1))
    np.save(f"{WORK}/tp_seed{seed}.npy", tp)
    np.save(f"{WORK}/ba_seed{seed}.npy", np.array([ba]))
    print(f"seed {seed}: OOF BA={ba:.5f}  ({time.time()-t0:.0f}s)", flush=True)

# hepsi hazirsa ortala ve submission yaz
if all(os.path.exists(f"{WORK}/tp_seed{s}.npy") for s in SEEDS):
    scores = [float(np.load(f"{WORK}/ba_seed{s}.npy")[0]) for s in SEEDS]
    tp_avg = np.mean([np.load(f"{WORK}/tp_seed{s}.npy") for s in SEEDS], axis=0)
    print(f"\nseed OOF ortalamasi: {np.mean(scores):.5f}  std: {np.std(scores):.5f}")
    print(f"exp3 (tek seed 42) idi: 0.94988")
    sub_pred = np.argmax(tp_avg / priors, axis=1)
    sub = pd.DataFrame({"id": test["id"], "health_condition": [inv_map[p] for p in sub_pred]})
    sub.to_csv(f"{WORK}/submission_seedavg.csv", index=False)
    np.save(f"{WORK}/test_proba_seedavg.npy", tp_avg)
    old = pd.read_csv(f"{DATA}/2-submission_features.csv")
    diff = (old["health_condition"].values != sub["health_condition"].values).mean()
    print(f"exp3 submission'dan farkli tahmin orani: {diff:.4%}")
    print("submission kaydedildi")
