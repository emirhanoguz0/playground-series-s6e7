# Deney 3: ozellik muhendisligi
# Baseline + esik ozellikleri (uretim kurali ipuclari):
#   sleep<6, sleep<7, stres/aktivite etkilesimleri, kural skoru
# Ayni CV bolmesi (seed=42) — baseline ile adil karsilastirma.
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

DATA = "/sessions/gifted-wizardly-sagan/mnt/playground-series-s6e7"
WORK = "/sessions/gifted-wizardly-sagan/work"
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

target_map = {"at-risk": 0, "unhealthy": 1, "fit": 2}
inv_map = {v: k for k, v in target_map.items()}
y = train["health_condition"].map(target_map).values

def add_features(df):
    out = df.copy()
    s = out["sleep_duration"]
    # esikler (NaN -> NaN kalir, native isleme devam)
    out["sleep_lt6"] = (s < 6).astype("float")
    out["sleep_lt7"] = (s < 7).astype("float")
    out.loc[s.isna(), ["sleep_lt6", "sleep_lt7"]] = np.nan
    # uretim kurali dogrudan: karar agacinin yolu
    stress = out["stress_level"]
    act = out["physical_activity_level"]
    # unhealthy kurali: sleep<6 & stress=high
    r_unh = ((s < 6) & (stress == "high")).astype("float")
    r_unh[s.isna() | stress.isna()] = np.nan
    out["rule_unhealthy"] = r_unh
    # fit kurali: sleep>=7 & stress=low & active
    r_fit = ((s >= 7) & (stress == "low") & (act == "active")).astype("float")
    r_fit[s.isna() | stress.isna() | act.isna()] = np.nan
    out["rule_fit"] = r_fit
    # stres x uyku bandi etkilesimi (kategorik)
    band = pd.cut(s, [-np.inf, 6, 7, np.inf], labels=["lt6", "6to7", "ge7"]).astype("object")
    inter = (stress.astype("object").fillna("na") + "_" + pd.Series(band, index=out.index).fillna("na"))
    out["stress_sleep"] = inter
    return out

feats_train = add_features(train)
feats_test = add_features(test)
feat_cols = [c for c in feats_train.columns if c not in ("id", "health_condition")]
X = feats_train[feat_cols]
X_test = feats_test[feat_cols]

for c in feat_cols:
    if not is_numeric_dtype(X[c]):
        X[c] = X[c].astype("category")
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)

params = dict(
    n_estimators=2000, learning_rate=0.05, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
    enable_categorical=True, early_stopping_rounds=100, random_state=SEED, n_jobs=-1,
)

priors = np.bincount(y) / len(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros((len(X), 3))
test_proba = np.zeros((len(X_test), 3))

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = XGBClassifier(**params)
    model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])
    test_proba += model.predict_proba(X_test) / 5
    ba = balanced_accuracy_score(y[va_idx], np.argmax(oof[va_idx] / priors, axis=1))
    print(f"fold {fold}: BA={ba:.5f}  best_iter={model.best_iteration}  ({time.time()-t0:.0f}s)", flush=True)

ba_prior = balanced_accuracy_score(y, np.argmax(oof / priors, axis=1))
print(f"\nOOF BA (prior correction): {ba_prior:.5f}")
print(f"baseline idi:              0.94975")
print(f"fark:                      {ba_prior-0.94975:+.5f}")

sub_pred = np.argmax(test_proba / priors, axis=1)
sub = pd.DataFrame({"id": test["id"], "health_condition": [inv_map[p] for p in sub_pred]})
sub.to_csv(f"{WORK}/submission_features.csv", index=False)
np.save(f"{WORK}/oof_features.npy", oof)
np.save(f"{WORK}/test_proba_features.npy", test_proba)
print(f"\ntoplam sure: {time.time()-t0:.0f}s")
