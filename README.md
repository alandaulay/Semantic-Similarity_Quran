# Project 4 — Quranic Semantic Similarity & Theme Discovery

Proyek ini menganalisis kemiripan semantik antarayat menggunakan embedding
multilingual Transformer, mengelompokkan ayat, lalu meminta Gemini merangkum
tema kelompok berdasarkan teks terjemahan yang diberikan. Hasil similarity
adalah kedekatan representasi model, bukan penetapan kesamaan makna atau tafsir.

## Data

Notebook mengambil dua edisi penuh dari Al Quran Cloud API:

- Arab: `quran-uthmani` — `https://api.alquran.cloud/v1/quran/quran-uthmani`
- Terjemahan Indonesia: `id.indonesian` — `https://api.alquran.cloud/v1/quran/id.indonesian`

Respons mentah disimpan apa adanya di `data/raw/`. Metadata mencatat edisi,
endpoint, waktu pengambilan, jumlah surah/ayat, dan status validasi. Dataset
terstruktur menggabungkan edisi berdasarkan nomor ayat global; validasi
membandingkan kelengkapan 114 surah/6.236 ayat serta identitas surah-ayat.

## Metodologi

Terjemahan Indonesia menjadi input embedding; teks Arab disimpan sebagai
referensi. Preprocessing hanya menormalkan Unicode dan spasi—tanpa stemming
atau penghapusan stopword. Model yang dipilih adalah
`paraphrase-multilingual-MiniLM-L12-v2`: model multilingual yang mendukung
bahasa Indonesia dan menghasilkan vektor 384 dimensi; ukurannya lebih ringan
daripada kandidat multilingual MPNet sehingga cocok untuk eksperimen lokal.
Notebook membandingkan kelayakan kedua kandidat secara kualitatif dan mencatat
bahwa pemilihan ini bukan klaim bahwa model paling akurat untuk terjemahan
Al-Qur'an. Nuansa penerjemahan, konteks ayat, dan hubungan tafsir tidak dapat
diwakili sempurna oleh embedding.

Cosine similarity dihitung pada embedding Transformer yang dinormalisasi,
bukan pada vektor TF-IDF. Nilai ini digunakan untuk pencarian top-5 tetangga,
pasangan global unik, serta ringkasan kedekatan cluster. Top-5 ayat sumber
diurutkan menurut rata-rata similarity lima tetangga terdekatnya; ini berbeda
dari lima pasangan global tertinggi. K-Means dievaluasi pada K=3..8 memakai
Silhouette (sampel maksimal 2.000 ayat), Davies-Bouldin, dan Calinski-Harabasz.
K=5 dipakai untuk memenuhi jumlah kelompok tugas, bukan diklaim otomatis
sebagai jumlah cluster yang paling koheren. Rata-rata cosine internal tiap
cluster dihitung terpisah dari ukuran cluster.

Gemini hanya menginterpretasikan tema; ia tidak menghitung similarity.
Analisis dilakukan satu cluster per permintaan dan prompt/respons disimpan.
Untuk menjaga ukuran permintaan, tiap cluster menyumbang sampel sistematis
maksimal 30 ayat yang diurutkan menurut similarity ke centroid. Sampel
mencakup rentang nilai similarity, tetapi tidak memeriksa seluruh anggota
cluster; hasil tema karena itu bersifat eksploratif. Tema bukan tafsir resmi
dan harus ditinjau kritis.

Repository ini hanya berisi kode dan dokumentasi. Dataset yang diunduh,
embedding, hasil eksperimen, grafik, dan prompt/respons tema tidak ikut
diunggah. Jalankan notebook untuk mengambil data dari API dan menghasilkan
artefak tersebut secara lokal.

## Instalasi

Gunakan Python 3.10 atau lebih baru. Dari folder proyek:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name quran-similarity --display-name "Python (Quran Similarity)"
```

Pastikan versi PyTorch yang dipasang sesuai dengan CPU/GPU yang tersedia.
Notebook memilih CUDA jika tersedia dan menjalankan CPU sebagai fallback.
Model embedding diunduh dari Hugging Face saat pertama kali dipakai dan
di-cache oleh Sentence Transformers.

## Konfigurasi Gemini

Folder `.env` pada workspace sudah berisi template `gemini.env.example`.
Salin menjadi `.env\gemini.env` dan isi `GEMINI_API_KEY` secara lokal.
Jika menjalankan hasil clone repository ini, salin `.env.example` menjadi
`.env` di folder proyek dan isi key secara lokal.
Notebook tidak menampilkan atau menyimpan key ke hasil eksperimen. Jika key
belum ada, tahapan tema mencatat status `blocked_configuration_required`;
tidak ada tema contoh atau hasil LLM rekaan yang disajikan.

## Menjalankan notebook

Buka `Project_4_Quranic_Semantic_Similarity.ipynb`, pilih kernel
`Python (Quran Similarity)`, lalu jalankan dari atas ke bawah. Unduhan API,
embedding, dan respons Gemini disimpan lokal untuk menghindari pemrosesan
ulang. Gunakan opsi `force_download` atau `force_recompute` di notebook hanya
jika memang ingin memperbarui cache.

## Struktur repository

```text
.
├── Project_4_Quranic_Semantic_Similarity.ipynb
├── quran_similarity.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Output

Folder berikut dibuat secara lokal setelah notebook dijalankan dan sengaja
tidak diunggah ke repository karena berisi data API, teks ayat, embedding,
atau keluaran eksperimen:

- `data/raw/`: respons API asli, metadata sumber, laporan validasi.
- `data/processed/`: CSV terstruktur dan teks hasil preprocessing.
- `data/embeddings/`: embedding `.npy` dan metadata model.
- `results/similarity/`: top-5 tiap ayat, top-5 ayat dengan tetangga terdekat,
  dan pasangan global tanpa duplikasi terbalik.
- `results/clustering/`: evaluasi K, label K=5, ukuran dan similarity internal.
- `results/theme_analysis/`: status analisis serta prompt dan respons aktual Gemini.
- `visualizations/`: grafik EDA, similarity, evaluasi cluster, dan tema (jika
  lima respons LLM aktual telah tersedia).

Seluruh angka dan grafik dihasilkan dari data API/model saat notebook berjalan.
Jika unduhan, validasi, model, atau API gagal, kegagalan dilaporkan dan tidak
diganti data sintetis.
