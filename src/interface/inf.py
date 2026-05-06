import sys
import os
import uuid
import random
import streamlit as st
from pathlib import Path

# Path Setup 
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR    = CURRENT_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

# Load library Supabase
try:
    from supabase import create_client, Client
except ImportError:
    st.error("Library 'supabase' belum diinstal. Tambahkan 'supabase' ke requirements.txt.")
    st.stop()

# Load ENV 
load_dotenv(BASE_DIR / ".env")

def _load_api_keys() -> list[str]:
    keys = []
    for var in ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3"]:
        val = st.secrets.get(var) if hasattr(st, "secrets") else None
        if not val:
            val = os.getenv(var)
        if val:
            keys.append(val)
    return keys

# Inisialisasi Supabase
@st.cache_resource
def init_supabase():
    url = st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") else os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

supabase = init_supabase()

def load_chat_history(session_id: str) -> list:
    if not supabase: return []
    try:
        response = supabase.table("chat_history").select("role, content").eq("session_id", session_id).order("id").execute()
        return response.data
    except Exception as e:
        st.sidebar.error(f"Supabase Load Error: {e}")
        return []

def save_chat_message(session_id: str, role: str, content: str):
    if not supabase: return
    try:
        supabase.table("chat_history").insert({"session_id": session_id, "role": role, "content": content}).execute()
    except Exception as e:
        st.sidebar.error(f"Supabase Save Error: {e}")

# Konstanta & Prompt 
DB_PATH = BASE_DIR / "chroma_db"
SUBTEST_NAMES = {
    "PU": "Penalaran Umum", "PPU": "Pengetahuan dan Pemahaman Umum", 
    "PBM": "Pemahaman Bacaan dan Menulis", "PK": "Pengetahuan Kuantitatif", 
    "PM": "Penalaran Matematika", "LBI": "Literasi Bahasa Indonesia", "LBE": "Literasi Bahasa Inggris"
}
MODEL_LIST = ["gemini-2.5-flash","gemini-3-flash-preview",]
SIMILARITY_THRESHOLD = 0.7

SYSTEM_PROMPT = r"""Kamu adalah AI Tutor pribadi yang ahli dan akurat untuk persiapan UTBK/SNBT.
Tugasmu membantu siswa berlatih soal berdasarkan konteks database.

ATURAN UTAMA:
1. Gunakan <konteks_database> sebagai referensi utama.
2. DILARANG menyalin ulang isi teks mentah dari database sebagai penjelasan materi.
3. Gunakan LaTeX untuk matematika: $...$ (inline) atau $$...$$ (blok). Jika bertingkat wajib gunakan \begin{aligned} ... \end{aligned}.
4. Jika membuat soal literasi, sertakan potongan teks bacaan HANYA di bagian teks soal.
5. DILARANG menulis kalimat proses berpikir seperti "Baik, saya akan...". Langsung eksekusi jawaban.

FORMAT GENERATE SOAL (SANGAT PENTING):
1. Setiap nomor soal WAJIB dipisahkan dengan dua baris baru agar tidak menumpuk.
2. Gunakan format Markdown berikut secara kaku:

## [NOMOR]. [JUDUL TOPIK]
**Soal:**
[Teks soal secara lengkap]

**Pilihan Jawaban:**
A. [Opsi A]
B. [Opsi B]
C. [Opsi C]
D. [Opsi D]
E. [Opsi E]

3. DILARANG menggabungkan teks soal dengan pilihan jawaban dalam satu paragraf.

ATURAN FORMAT MATEMATIKA (SANGAT PENTING):
1. Semua angka, variabel, rumus, dan simbol matematika WAJIB ditulis dalam format LaTeX.
2. Rumus sebaris (inline): Gunakan $...$
3. Rumus blok (display): Gunakan $$...$$
4. Rumus bertingkat WAJIB menggunakan \begin{aligned} ... \end{aligned}.
5. WAJIB meletakkan tanda $$ secara presisi sebelum \begin{aligned} dan sesudah \end{aligned}.
6. DILARANG menulis \begin{aligned} tanpa tanda $$ di luarnya.

<konteks_database>
{context}
</konteks_database>

Pertanyaan Siswa: {user_query}
"""

# Inisialisasi Resource
@st.cache_resource
def _init_collection():
    # Set cache dir agar model tidak re-download tiap cold start di Streamlit Cloud
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME", str(BASE_DIR / "models" / "st_cache"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)

    model_folder = BASE_DIR / "models" / "e5-small"
    model_id = str(model_folder) if model_folder.exists() else "intfloat/multilingual-e5-small"
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_id)
    client_chroma = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client_chroma.get_collection(name="ai_tutor_collection", embedding_function=ef)
    return collection, "Lokal" if model_folder.exists() else "Download"

@st.cache_resource
def _init_gemini():
    keys = _load_api_keys()
    if not keys: return []
    return [genai.Client(api_key=k) for k in keys]

def init_resources():
    gemini_clients = _init_gemini()
    collection, embed_source = _init_collection()
    return gemini_clients, collection, embed_source

# ── Retrieval Logic (Randomized & Context Aware) ──────────────────────────────
def retrieve_context(collection, query: str, subtest_filter=None) -> str:
    # 1. Deteksi referensi riwayat (Context Aware)
    history_ref = ["chat diatas", "tadi", "sebelumnya", "soal itu", "materi yang sama"]
    if any(ref in query.lower() for ref in history_ref):
        last_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        # Gunakan index -2 karena index -1 adalah input user yang baru saja disubmit
        if len(last_msgs) > 1:
            query_sebelumnya = last_msgs[-2]
            query = f"{query_sebelumnya} {query}"

    # 2. Batas Penarikan Dinamis
    is_literasi = subtest_filter in ["LBI", "LBE", "PPU", "PBM"]
    target_n = 2 if is_literasi else 3
    
    # 3. Strategi Shuffle
    sample_size = target_n * 4
    where_clause = {"subtest": subtest_filter} if subtest_filter else None

    # Clamp sample_size agar tidak melebihi jumlah dokumen yang tersedia di collection
    try:
        total_docs = collection.count()
        sample_size = min(sample_size, max(total_docs, 1))
    except Exception:
        pass  # Biarkan query berjalan dengan sample_size asli jika count gagal
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=sample_size,
            where=where_clause
        )
    except Exception as e:
        st.sidebar.error(f"Pencarian Database Error: {e}")
        return ""
    
    if not results["documents"] or not results["documents"][0]:
        return ""
        
    kandidat = results["documents"][0]
    random.shuffle(kandidat) 
    
    selected_docs = kandidat[:target_n]
    return "\n\n---\n\n".join(selected_docs)

# API Model & Prompt Logic 
def ask_gemini(clients, prompt):
    if not clients: return "⚠️ Konfigurasi API Key tidak ditemukan."
    
    for client in clients:
        for model in MODEL_LIST:
            try:
                res = client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=5000, temperature=0.7)
                )
                return res.text
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "exhausted" in err:
                    break  # Quota habis → langsung lompat ke API Key berikutnya
                if "400" in err or "404" in err:
                    continue  # Model tidak valid → coba model berikutnya di key yang sama
                if "403" in err:
                    break  # Unauthorized → lompat ke API Key berikutnya
                continue  # Error lain → coba model berikutnya
                
    return "⚠️ Semua API Key Limit atau tidak tersedia."

def build_prompt(mode, phase, context, history, user_input, soal_aktif):
    history_text = "\n".join([f"{'Siswa' if m['role']=='user' else 'Tutor'}: {m['content']}" for m in history[-6:]])
    
    if mode == "Latihan Soal":
        if phase == "generate":
            instruksi = "Buat soal latihan baru dari database. JANGAN ulangi topik/soal yang sudah ada di [RIWAYAT] bawah ini. Tampilkan soal saja tanpa kunci jawaban."
        else:
            context = f"SOAL AKTIF: {soal_aktif}\n{context}"
            instruksi = "Evaluasi jawaban siswa secara ringkas. Format: Status (Benar/Salah) -> Pembahasan -> Kunci."
    else:
        instruksi = "Jawab pertanyaan siswa secara ringkas."

    # MENGGUNAKAN REPLACE AGAR TIDAK ERROR DENGAN LATEX {aligned}
    filled_prompt = SYSTEM_PROMPT.replace("{context}", context if context else "N/A").replace("{user_query}", user_input)
    
    return f"{filled_prompt}\n\n[RIWAYAT]\n{history_text}\n\n[PERINTAH]\n{instruksi}"

# Main Application 
def main():
    st.set_page_config(page_title="AI Tutor UTBK", layout="wide")
    
    with st.spinner("Memastikan koneksi database dan AI siap..."):
        clients, collection, embed_source = init_resources()
        if "system_ready" not in st.session_state:
            try:
                collection.count()
            except:
                pass
            st.session_state.system_ready = True

    if "messages" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = load_chat_history(st.session_state.session_id)
        st.session_state.latihan_phase = "generate"
        st.session_state.soal_aktif = None

    with st.sidebar:
        st.title("📚 AI Tutor")
        mode = st.radio("Mode", ["Chat Materi", "Latihan Soal"],
                        index=["Chat Materi", "Latihan Soal"].index(
                            st.session_state.get("mode", "Chat Materi")
                        ))
        # Simpan mode ke session_state; reset phase jika mode berpindah
        if st.session_state.get("mode") != mode:
            st.session_state.mode = mode
            st.session_state.latihan_phase = "generate"
            st.session_state.soal_aktif = None
        sub_idx = st.selectbox("Subtest", range(len(SUBTEST_NAMES)+1), 
                               format_func=lambda i: (["Semua"]+list(SUBTEST_NAMES.keys()))[i])
        sub_filter = (["None"]+list(SUBTEST_NAMES.keys()))[sub_idx]
        sub_filter = None if sub_filter == "None" else sub_filter
        
        with st.expander("🛠️ Debug Status", expanded=False):
            st.write(f"**API Keys:** {len(clients)} aktif")
            st.write(f"**Embedding:** {embed_source}")
            st.write(f"**Supabase:** {'Online' if supabase else 'Offline'}")
        
        if st.button("🗑️ Sesi Baru", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if user_input := st.chat_input("Tanya Nay..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_chat_message(st.session_state.session_id, "user", user_input)

        with st.chat_message("assistant"):
            with st.spinner("Memproses..."):
                ctx = ""
                if not (mode == "Latihan Soal" and st.session_state.latihan_phase == "evaluasi"):
                    ctx = retrieve_context(collection, user_input, sub_filter)
                
                prompt = build_prompt(mode, st.session_state.latihan_phase, ctx, 
                                     st.session_state.messages, user_input, st.session_state.soal_aktif)
                ans = ask_gemini(clients, prompt)
                st.markdown(ans)
        
        st.session_state.messages.append({"role": "assistant", "content": ans})
        save_chat_message(st.session_state.session_id, "assistant", ans)

        if mode == "Latihan Soal":
            if st.session_state.latihan_phase == "generate":
                st.session_state.soal_aktif = ans
                st.session_state.latihan_phase = "evaluasi"
            else:
                st.session_state.latihan_phase = "generate"
            st.rerun()

if __name__ == "__main__":
    main()
