**Laporan Proyek Machine Learning - Dimas Ananta Kusuma/231113007**

**Project Domain**
Industri hiburan digital menghadapi tantangan Information Overload, di mana pengguna kesulitan memilih film dari ribuan katalog. Sistem rekomendasi menjadi solusi krusial untuk mempersonalisasi saran film bagi pengguna guna meningkatkan kepuasan dan retensi pada platform platform streaming.

**Business Understanding**
Problem Statements
-Bagaimana cara merekomendasikan film yang serupa dengan film yang pernah ditonton pengguna berdasarkan karakteristik genre?
-Seberapa akurat algoritma Matrix Factorization (SVD) dalam memprediksi rating film untuk pengguna?

Goals
-Menghasilkan daftar rekomendasi film yang relevan menggunakan pendekatan Content-Based Filtering.
-Mengembangkan model Collaborative Filtering dengan tingkat kesalahan prediksi (RMSE) di bawah 1.0.

Solution statements
-Solution 1 (Content-Based): Menggunakan TF-IDF Vectorizer untuk ekstraksi fitur genre dan Cosine Similarity untuk menghitung derajat kemiripan antar film.
-Solution 2 (Collaborative): Menggunakan algoritma SVD (Singular Value Decomposition) yang mampu menangkap pola tersembunyi (latent factors) dari interaksi pengguna dan film.

**Data Understanding**
Dataset yang digunakan adalah MovieLens Dataset (versi small) yang mencakup beberapa berkas utama:
-movie.csv: Data profil film (62.423 baris) berisi movieId, title, dan genres.
-rating.csv: Interaksi pengguna (25.000.095 baris) berisi userId, movieId, rating, dan timestamp. (Digunakan subset 100.000 baris).
-Berkas Lainnya: tags.csv, links.csv, genome-scores.csv, dan genome-tags.csv yang menyediakan metadata tambahan.

Variabel Utama:
-genres: Fitur kategori film untuk Content-Based Filtering.
-rating: Target prediksi (skala 0.5 - 5.0) untuk Collaborative Filtering.

**Data Preparation**
-Data Selection: Memilih file movie.csv dan rating.csv sebagai sumber data utama.
-Data Cleaning: Menghapus nilai kosong (missing values) dan data duplikat untuk menjaga kualitas model.
-Feature Engineering: Membersihkan string pada kolom genre agar dapat diproses oleh vectorizer.
-Data Splitting: Membagi data rating menjadi 80% data latih dan 20% data uji untuk evaluasi model SVD.

**Modeling**
1. Content-Based Filtering
Model ini merekomendasikan film berdasarkan kemiripan konten (genre).

Python
# Ekstraksi fitur genre menggunakan TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Menghitung similarity score
cosine_sim = cosine_similarity(tfidf_matrix)
Kelebihan: Dapat merekomendasikan film baru yang belum memiliki rating (item-side cold start).

Kekurangan: Tidak dapat memberikan rekomendasi di luar kategori genre yang sudah dikenal pengguna.

2. Collaborative Filtering (SVD)
Model ini belajar dari perilaku kolektif seluruh pengguna untuk memberikan saran personal.

Python

# Melatih model SVD
algo = SVD()
algo.fit(trainset)
Kelebihan: Mampu menangkap selera pengguna yang kompleks melampaui sekadar genre.

Kekurangan: Mengalami kesulitan saat menghadapi pengguna baru yang belum memberikan rating (user-side cold start).

**Evaluation**
Metrik evaluasi yang digunakan adalah RMSE (Root Mean Squared Error), yang mengukur rata-rata penyimpangan antara rating prediksi dan rating aktual.

Python

# Evaluasi performa model
predictions = algo.test(testset)
accuracy.rmse(predictions)
Hasil Evaluasi:

Model mencapai nilai RMSE: 0.9157.

Interpretasi: Dengan nilai RMSE di bawah 1.0, akurasi model tergolong sangat baik. Rata-rata selisih prediksi model dengan rating asli hanya sebesar ~0.9 poin pada skala rating 1-5, sehingga rekomendasi yang diberikan sangat mendekati preferensi asli pengguna.
