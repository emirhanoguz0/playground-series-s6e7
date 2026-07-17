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

| # | Deney | OOF CV | Public LB | Submission |
|---|---|---|---|---|
| — | Baseline XGB, düz argmax | 0.87852 | — | — |
| 1 | Baseline XGB + prior correction | 0.94975 | 0.94979 | `1-submission_baseline.csv` |
| — | + karar katsayısı optimizasyonu (exp2) | 0.94983 | denenmedi | — |
| 2 | + eşik özellikleri: sleep<6/7, kural, etkileşim (exp3) | 0.94988 | 0.95004 | `2-submission_features.csv` |
| — | LightGBM (aynı özellikler) | 0.94959 | — | — |
| — | CatBoost (aynı özellikler) | 0.94918 | — | — |
| 3 | Blend: 0.8×XGB + 0.1×LGBM + 0.1×CatBoost (exp4) | **0.94993** | 0.95001 | `3-submission_blend.csv` |
| — | Sadece 3 çekirdek özellik + türetilmişler (exp5) | 0.94392 | — | — |
| 4 | Seed ortalaması: 5 seed × exp3 modeli (exp6) | 0.94990* | 0.95000 | `4-submission_seedavg.csv` |

\* exp6 CV'si tek OOF değil, 5 seed'in OOF ortalaması (42/7/123/2024/555:
0.94988/0.94991/0.94990/0.94987/0.94995, std 0.00003).

CV ile LB farkları 0.0002'nin altında — CV altyapısı güvenilir.

Notlar:
- exp2 (karar katsayısı taraması) sadece +0.00008 verdi — gürültü
  seviyesinde, OOF'a ezber riski nedeniyle kullanılmadı.
- exp3 eşik özellikleri 5 fold'un 4'ünde iyileşme sağladı (+0.00013).

Baseline fold skorları: 0.95042 / 0.95183 / 0.94893 / 0.94953 / 0.94802
exp3 fold skorları: 0.95054 / 0.95194 / 0.94937 / 0.94946 / 0.94811

## Çalıştırma

```bash
pip install -r requirements.txt
python baseline_xgb.py
```

Veri dosyaları (`train.csv`, `test.csv`) Kaggle'dan indirilip
proje kök klasörüne konmalı. Eğitim CPU'da ~15 dk sürüyor.

## Yol haritası

- [x] Baseline XGB + native NaN + prior correction (CV 0.94975, LB 0.94979)
- [x] Karar katsayısı optimizasyonu — gürültü seviyesi, kullanılmadı
- [x] Özellik mühendisliği: sleep<6, sleep<7, kural özellikleri (CV 0.94988, LB 0.95004)
- [x] Model çeşitlendirme: LightGBM (0.94959), CatBoost (0.94918)
- [x] Blend: 0.8/0.1/0.1 ağırlıklarla CV 0.94993 (LB 0.95001) — LB'de exp3'ün gerisinde
- [x] exp5: sadece 3 çekirdek özellik — CV 0.94392, büyük düşüş. "Gürültü" kolonları
      eksik değer kalıpları üzerinden sinyal taşıyor, atılmamalı.
- [x] exp6: seed ortalaması (5 seed) — seed OOF ort. 0.94990, LB 0.95000.
      exp3'ü (0.95004) geçemedi; kazançlar artık gürültü içinde kayboluyor.
