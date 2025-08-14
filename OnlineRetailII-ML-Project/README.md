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


## Lisensi
MIT
