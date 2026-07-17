# Deney 4: model cesitlendirme + blend
# LGBM ve CatBoost'u exp3 ozellikleriyle egit (ayni CV bolmesi, seed=42).
# XGB OOF'u exp3'ten hazir. Uc modelin olasiliklarini blend et,
# agirliklari OOF uzerinde ara, prior correction uygula.
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

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

cat_cols = [c for c in feat_cols if not is_numeric_dtype(X[c])]
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)

# CatBoost: NaN kategorikleri string ister
Xc = X.copy()
Xc_test = X_test.copy()
for c in cat_cols:
    Xc[c] = Xc[c].astype("object").fillna("NA").astype(str)
    Xc_test[c] = Xc_test[c].astype("object").fillna("NA").astype(str)

priors = np.bincount(y) / len(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(skf.split(X, y))

def run_cv(name, make_model, Xd, Xd_test, fit_kwargs_fn):
    oof = np.zeros((len(Xd), 3))
    tp = np.zeros((len(Xd_test), 3))
    t0 = time.time()
    for fold, (tr, va) in enumerate(splits):
        m = make_model()
        m.fit(Xd.iloc[tr], y[tr], **fit_kwargs_fn(Xd.iloc[va], y[va]))
        oof[va] = m.predict_proba(Xd.iloc[va])
        tp += m.predict_proba(Xd_test) / 5
        ba = balanced_accuracy_score(y[va], np.argmax(oof[va] / priors, axis=1))
        print(f"{name} fold {fold}: BA={ba:.5f} ({time.time()-t0:.0f}s)", flush=True)
    ba = balanced_accuracy_score(y, np.argmax(oof / priors, axis=1))
    print(f"{name} OOF BA: {ba:.5f}\n", flush=True)
    np.save(f"{WORK}/oof_{name}.npy", oof)
    np.save(f"{WORK}/test_proba_{name}.npy", tp)
    return oof, tp

# LightGBM
lgb_oof, lgb_tp = run_cv(
    "lgbm",
    lambda: LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="multiclass", num_class=3,
        random_state=SEED, n_jobs=-1, verbose=-1),
    X, X_test,
    lambda Xv, yv: dict(eval_set=[(Xv, yv)], callbacks=[early_stopping(100, verbose=False)]),
)

# CatBoost
cb_oof, cb_tp = run_cv(
    "catboost",
    lambda: CatBoostClassifier(
        iterations=3000, learning_rate=0.08, depth=6,
        loss_function="MultiClass", cat_features=cat_cols,
        random_seed=SEED, early_stopping_rounds=100, verbose=0),
    Xc, Xc_test,
    lambda Xv, yv: dict(eval_set=(Xv, yv)),
)

# XGB exp3'ten
xgb_oof = np.load(f"{WORK}/oof_features.npy")
xgb_tp = np.load(f"{WORK}/test_proba_features.npy")

# blend agirligi arama (grid, toplam=1)
best = (0, None)
for wx in np.arange(0, 1.01, 0.1):
    for wl in np.arange(0, 1.01 - wx, 0.1):
        wc = 1 - wx - wl
        blend = wx * xgb_oof + wl * lgb_oof + wc * cb_oof
        ba = balanced_accuracy_score(y, np.argmax(blend / priors, axis=1))
        if ba > best[0]:
            best = (ba, (wx, wl, wc))
print(f"en iyi blend: BA={best[0]:.5f}  w(xgb,lgbm,cat)={np.round(best[1],2)}")

wx, wl, wc = best[1]
blend_tp = wx * xgb_tp + wl * lgb_tp + wc * cb_tp
sub_pred = np.argmax(blend_tp / priors, axis=1)
sub = pd.DataFrame({"id": test["id"], "health_condition": [inv_map[p] for p in sub_pred]})
sub.to_csv(f"{WORK}/3-submission_blend.csv", index=False)
print("submission kaydedildi")
