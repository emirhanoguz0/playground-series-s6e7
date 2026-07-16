# Deney 2a: karar katsayisi optimizasyonu
# Egitim yok — baseline'in OOF olasiliklari uzerinde arama.
# Mantik: pred = argmax(proba * w). w'yi CV skoru maksimum olacak sekilde bul.
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from itertools import product

DATA = "/sessions/gifted-wizardly-sagan/mnt/playground-series-s6e7"
WORK = "/sessions/gifted-wizardly-sagan/work"

train = pd.read_csv(f"{DATA}/train.csv", usecols=["health_condition"])
target_map = {"at-risk": 0, "unhealthy": 1, "fit": 2}
y = train["health_condition"].map(target_map).values
oof = np.load(f"{WORK}/oof_baseline.npy")

priors = np.bincount(y) / len(y)
base = balanced_accuracy_score(y, np.argmax(oof / priors, axis=1))
print(f"referans (1/prior): {base:.5f}")

# 1/prior etrafinda carpan taramasi (at-risk sabit=1, digerleri serbest)
w0 = 1.0 / priors
best = (base, w0 / w0[0])
for m1, m2 in product(np.linspace(0.6, 1.6, 21), repeat=2):
    w = np.array([1.0, w0[1] / w0[0] * m1, w0[2] / w0[0] * m2])
    ba = balanced_accuracy_score(y, np.argmax(oof * w, axis=1))
    if ba > best[0]:
        best = (ba, w)
print(f"kaba tarama en iyi: {best[0]:.5f}  w={np.round(best[1],3)}")

# ince tarama
w_c = best[1]
for m1, m2 in product(np.linspace(0.94, 1.06, 25), repeat=2):
    w = np.array([1.0, w_c[1] * m1, w_c[2] * m2])
    ba = balanced_accuracy_score(y, np.argmax(oof * w, axis=1))
    if ba > best[0]:
        best = (ba, w)
print(f"ince tarama en iyi:  {best[0]:.5f}  w={np.round(best[1],3)}")
print(f"kazanc: {best[0]-base:+.5f}")
np.save(f"{WORK}/best_weights.npy", best[1])
