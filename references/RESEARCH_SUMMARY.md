# GeoESG Research References (2024-2026)

Dokumen ini berisi daftar paper penelitian yang menjadi landasan ilmiah pengembangan fitur GeoESG v2.3.

## 1. Machine Learning Biomass Estimation (XGBoost & Fusion)
* **Zhang et al. (2025)** - *High-resolution mapping of forest parameters in tropical rainforests through AutoML integration of GEDI with Sentinel-1/2, Landsat 8 and ALOS-2 data*. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
  * **Penerapan**: Fusi multi-sensor (Optik + C-Band + L-Band) dan penggunaan Gradient Boosting.
* **Wang et al. (2024)** - *Estimation of aboveground biomass for different forest types using data from Sentinel-1, Sentinel-2, ALOS PALSAR-2, and GEDI*. Forests, 15(9), 1576.
  * **Penerapan**: Komparasi algoritma, membuktikan GBRT/XGBoost superior di beberapa skenario dibanding Random Forest.
* **Lee et al. (2026)** - *Direct mapping of plantation forest aboveground biomass change with deep learning and SAR-optical fusion*. EarthArXiv.
  * **Penerapan**: Fusi Sentinel-1, ALOS PALSAR L-band, dan Sentinel-2.

## 2. Tree Crown Detection (Deep Learning)
* **Brandt et al. (2025)** - *High-resolution sensors and deep learning models for tree resource monitoring*. Nature Reviews Earth & Environment.
  * **Penerapan**: Landasan transisi dari computer vision klasik (HSV thresholding) ke deep learning (segmentasi morfologi / instance segmentation).
* **Deepalakshmi & Thenmalar (2026)** - *NDVI-Enhanced Multisensor Tree Detection using YOLOv9 and U-Net*. IEEE.
  * **Penerapan**: Arsitektur hybrid untuk deteksi tajuk pohon.

## 3. Carbon Accounting & ESG Assurance
* **Wijayanto et al. (2026)** - *AI-Driven Carbon Pricing Optimization: A Geospatial Analysis Framework for Indonesia's Energy Transition*. Indonesian Journal of Energy.
  * **Penerapan**: Integrasi harga karbon (IDXCarbon) dengan MRV spasial untuk konteks transisi energi Indonesia.
* **Gizachew (2026)** - *Artificial Intelligence and Machine Learning in Remote Sensing for Tropical Forest Monitoring*. Remote Sensing.
  * **Penerapan**: Framework Digital MRV (D-MRV) untuk REDD+.
* **Garg (2024)** - *CNN-based image validation for ESG reporting: An explainable AI and blockchain approach*. Int. J. Comput. Sci. Inf. Technol. Res.
  * **Penerapan**: Penggunaan citra satelit sebagai instrumen anti-greenwashing pada laporan ESG.
