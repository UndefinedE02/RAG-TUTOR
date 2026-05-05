# 📚 AI Tutor UTBK/SNBT

Aplikasi tutor pribadi berbasis AI untuk persiapan **UTBK/SNBT** yang dibangun di atas arsitektur **RAG (Retrieval-Augmented Generation)**. Sistem ini mengkombinasikan vector database lokal dengan model LLM Gemini untuk menghasilkan jawaban yang akurat, kontekstual, dan meminimalisir halusinasi.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **Chat Materi** | Tanya konsep, minta penjelasan, diskusi soal secara bebas |
| **Latihan Soal** | Generate 5 soal baru berdasarkan referensi database secara otomatis |
| **Evaluasi Jawaban** | Evaluasi jawaban siswa dengan pembahasan lengkap per nomor |
| **Filter Subtest** | PU, PPU, PBM, PK, PM, LBI, LBE |
| **Persistensi Sesi** | Riwayat chat tersimpan via Supabase, tidak hilang saat halaman di-refresh |
| **Retrieval Cerdas** | Randomisasi chunk + context-aware query untuk variasi soal maksimal |
| **Fault Tolerance** | Rotasi API key otomatis + exponential backoff saat rate limit |

---

## 🏗️ Arsitektur

```
PDF Materi
    │
    ├── igs.py          → Parsing & chunking teks
    │
    └── emd.py          → Embedding → ChromaDB (Vector DB)
                                │
                    User Input (Streamlit UI)
                                │
                        retrieve_context()
                        (Similarity Search + Shuffle)
                                │
                          build_prompt()
                          (Konteks + Riwayat + Instruksi)
                                │
                          ask_gemini()
                          (Rotasi Key + Backoff)
                                │
                           Jawaban / Soal
                                │
                           Supabase (Simpan Riwayat)
```

---

## 📂 Struktur Folder

```
AiTutor/
├── chroma_db/                  ← Vector database (auto-generated)
├── data/
│   ├── pdf/                    ← Taruh semua file PDF di sini
│   └── chunks.pkl              ← Hasil parsing (auto-generated)
├── models/
│   └── e5-small/               ← Model embedding lokal (opsional)
├── llm_env/                    ← Virtual environment Python
├── src/
│   ├── interface/
│   │   └── inf.py              ← Aplikasi utama (Streamlit)
│   ├── ingestion/
│   │   ├── igs.py              ← Parsing PDF → chunks.pkl
│   │   └── emd.py              ← chunks.pkl → ChromaDB
│   └── retrieval/
│       ├── rtl.py              ← Test retrieval via terminal
│       └── cekdb.py            ← Diagnostik kondisi database
├── main.py                     ← Alternatif antarmuka terminal (CLI)
├── .env                        ← API key (buat manual, jangan di-commit)
├── requirements.txt
└── README.md
```

---

## ⚙️ Prasyarat

- Python **3.10+**
- Koneksi internet (Gemini API)
- **Google Gemini API key** — daftar gratis di [Google AI Studio](https://aistudio.google.com/apikey)
- **Supabase project** (opsional, untuk persistensi riwayat chat)

---

## 🚀 Instalasi

### 1. Clone repository
```bash
git clone https://github.com/UndefinedE02/AI-TUTOR.git
cd AI-TUTOR
```

### 2. Buat dan aktifkan virtual environment
```bash
python3 -m venv llm_env
source llm_env/bin/activate        # Linux/macOS
llm_env\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Buat file `.env` di root project
```env
# Wajib
GOOGLE_API_KEY=AIzaSy_xxxxxxxxxxxxxxx

# Opsional — untuk rotasi key saat rate limit
GOOGLE_API_KEY_2=AIzaSy_xxxxxxxxxxxxxxx
GOOGLE_API_KEY_3=AIzaSy_xxxxxxxxxxxxxxx

# Opsional — untuk persistensi riwayat chat (Supabase)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=eyJxxxxxxxxxxxxxxx
```

> **Catatan:** Jika deploy ke Streamlit Cloud, masukkan semua variabel di atas melalui dashboard **Settings → Secrets**, bukan file `.env`.

### 5. Masukkan file PDF ke `data/pdf/`

Nama file **wajib** mengandung kode subtest agar terdeteksi otomatis:
```
to_PK_2024.pdf
[PEMBAHASAN-PM]_soal1.pdf
latihan_LBI_py3.pdf
```

### 6. Install model embedding (opsional, untuk mode offline)

Unduh model `intfloat/multilingual-e5-small` dan letakkan di `models/e5-small/`. Jika folder tidak ada, aplikasi akan download otomatis saat pertama kali dijalankan.

### 7. Setup Supabase (opsional)

Buat tabel `chat_history` di project Supabase kamu dengan skema berikut:
```sql
create table chat_history (
  id bigint generated always as identity primary key,
  session_id text not null,
  role text not null,
  content text not null,
  created_at timestamptz default now()
);
```

---

## ▶️ Menjalankan Aplikasi

### Langkah 1 — Parsing PDF
```bash
python src/ingestion/igs.py
```

### Langkah 2 — Build vector database
```bash
python src/ingestion/emd.py
```
> Download model embedding (~470MB) hanya terjadi sekali di langkah ini.

### Langkah 3 — Jalankan aplikasi
```bash
streamlit run src/interface/inf.py
```
Buka browser di `http://localhost:8501`

### Alternatif — Antarmuka terminal (CLI)
```bash
python main.py
```

---

## 🔍 Utilitas Database

```bash
# Cek jumlah chunk per subtest dan kondisi database
python src/retrieval/cekdb.py

# Test retrieval manual via terminal
python src/retrieval/rtl.py
```

---

## 📖 Subtest yang Didukung

| Kode | Nama |
|---|---|
| PU | Penalaran Umum |
| PPU | Pengetahuan dan Pemahaman Umum |
| PBM | Pemahaman Bacaan dan Menulis |
| PK | Pengetahuan Kuantitatif |
| PM | Penalaran Matematika |
| LBI | Literasi Bahasa Indonesia |
| LBE | Literasi Bahasa Inggris |

---

## 📝 Catatan Penting

- PDF hasil scan (gambar) tidak dapat diproses — gunakan PDF dengan teks yang bisa di-select
- Nama file PDF harus mengandung kode subtest agar pipeline ingestion mendeteksi dengan benar
- Database demo yang tersedia saat ini berisi **±5.700 chunk dari 250+ file PDF**
- Untuk pembelian database lengkap, hubungi pembuat project

---

## 📄 Lisensi

Project ini dibuat untuk keperluan edukasi. Penggunaan ulang dan modifikasi diperbolehkan dengan menyertakan kredit.
