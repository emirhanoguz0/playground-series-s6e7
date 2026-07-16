# Kaggle Playground S6E7 — Predicting Student Health Risk

Öğrenci sağlık riski tahmini. 3 sınıf: `fit`, `at-risk`, `unhealthy`.
Metrik: balanced accuracy.

## Veri

- train: 690.088 satır, 13 özellik (7 sayısal + 6 kategorik)
- test: 295.753 satır
- Sınıf dağılımı çok dengesiz: %86 at-risk, %8 unhealthy, %6 fit
- Her kolonda %1–12 eksik değer var

## Yarışma araştırması (önemli bulgular)

1. Orijinal veri deterministik bir kuralla üretilmiş (derinlik-4 karar ağacı).
   Sadece 3 özellik belirleyici: `sleep_duration` (6 ve 7 eşikleri),
   `stress_level`, `physical_activity_level`.
2. Eksik değerleri doldurmak (median/mode) sinyali bozuyor.
   XGBoost'un native NaN işleme özelliği kullanılmalı (LB: 0.903 → 0.950 farkı).
3. Public LB test verisinin sadece %20'si. CV'ye güven, LB'ye değil.
4. Skor tavanı ~0.951. Üstü çoğunlukla gürültü.

## Yaklaşım (baseline_xgb.py)

- **Model:** XGBoost, `enable_categorical=True`, eksikler doldurulmadan (native NaN)
- **CV:** StratifiedKFold(5, shuffle, seed=42), out-of-fold balanced accuracy
- **Dengesizlik çözümü:** prior correction — tahmin olasılıkları sınıf
  oranlarına bölünüp argmax alınıyor. Düz argmax 0.879 verirken bu 0.950 veriyor.
- **Early stopping:** 100 round, fold başına ~500 ağaçta duruyor

## Sonuçlar

| Deney | OOF CV | Public LB |
|---|---|---|
| Baseline XGB, düz argmax | 0.87852 | — |
| Baseline XGB + prior correction | **0.94975** | **0.94979** |

CV ile LB farkı sadece 0.00004 — CV altyapısı güvenilir.

Fold skorları: 0.95042 / 0.95183 / 0.94893 / 0.94953 / 0.94802

## Çalıştırma

```bash
pip install -r requirements.txt
python baseline_xgb.py
```

Veri dosyaları (`train.csv`, `test.csv`) Kaggle'dan indirilip
proje kök klasörüne konmalı. Eğitim CPU'da ~15 dk sürüyor.

## Yol haritası

- [x] Baseline XGB + native NaN + prior correction (CV 0.94975)
- [ ] Özellik mühendisliği: sleep<6, sleep<7 eşik özellikleri, stres etkileşimi
- [ ] Model çeşitlendirme: LightGBM, CatBoost
- [ ] Blend / ensemble (hedef ~0.951)
