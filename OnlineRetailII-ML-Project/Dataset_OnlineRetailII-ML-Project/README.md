# OnlineRetailII-ML-Project

**Tujuan Proyek**
1. **EDA Penjualan**: ringkas tren pendapatan bulanan, produk terlaris, dan negara penyumbang revenue.
2. **Segmentasi Pelanggan (RFM + K-Means)**: kelompokkan pelanggan berdasarkan *Recency*, *Frequency*, dan *Monetary* untuk strategi pemasaran.
3. **Prediksi Nilai Invoice (Regresi)**: memprediksi total nilai transaksi per invoice dengan RandomForestRegressor.

**Algoritma yang Digunakan & Alasan**
- **K-Means** (segmentasi RFM): cepat, mudah diinterpretasi, efektif untuk klasterisasi ketika fitur telah dinormalisasi/log-transform. Pemilihan jumlah klaster dilakukan otomatis dengan **silhouette score**.
- **RandomForestRegressor** (prediksi): kuat terhadap outlier/nonlinearitas, minim *feature engineering* untuk baseline yang bagus, memberikan *feature importance* untuk interpretasi.
- **Statistik Deskriptif + Resampling bulanan** (EDA): untuk memahami pola bisnis sebelum modeling.

---

## Struktur Proyek
```
OnlineRetailII-ML-Project/
├── src/
│   └── online_retail_II_analysis.py
├── outputs/            # hasil (grafik PNG, CSV, JSON) akan muncul di sini setelah dijalankan
├── notebooks/          # opsional jika ingin membuat notebook sendiri
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Cara Menjalankan (Lokal)
1. Pastikan Python 3.10+ terpasang.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan skrip (ganti path sesuai lokasi file Anda):
   ```bash
   python src/online_retail_II_analysis.py --data_path "006_SMTR6/MACHINE LEARNING/online_retail_II.xlsx" --out_dir "outputs"
   ```
4. Hasil akan tersimpan di folder `outputs/`:
   - `monthly_revenue.png`
   - `top10_products.png`
   - `top10_countries.png`
   - `rfm_scatter.png`, `rfm_segments.csv`, `rfm_meta.json`
   - `feature_importance.png`, `regression_metrics.json`, `predictions.csv`

## Cara Menjalankan (Google Colab + Google Drive PATH)
```python
from google.colab import drive
drive.mount('/content/drive')

# Ganti path sesuai folder Drive Anda
data_path = "/content/drive/MyDrive/006_SMTR6/MACHINE LEARNING/online_retail_II.xlsx"

!pip install -r "/content/OnlineRetailII-ML-Project/requirements.txt"
!python "/content/OnlineRetailII-ML-Project/src/online_retail_II_analysis.py" --data_path "$data_path" --out_dir "/content/OnlineRetailII-ML-Project/outputs"
```

> **Catatan**: Jika file Anda versi `.csv`, cukup ganti argumen `--data_path` ke file `.csv` tersebut.

## Ekspor ke GitHub (Contoh)
```bash
# di dalam folder OnlineRetailII-ML-Project
git init
git add .
git commit -m "Online Retail II - E2E ML analysis (EDA, RFM KMeans, RF Regression)"
# buat repo baru di GitHub lalu tambahkan remote Anda
git remote add origin https://github.com/<username>/OnlineRetailII-ML-Project.git
git branch -M main
git push -u origin main
```

## Output Visual (berwarna)
Seluruh grafik dibuat dengan **matplotlib** dan akan otomatis berwarna (default *color cycle*).

## Lisensi
MIT
