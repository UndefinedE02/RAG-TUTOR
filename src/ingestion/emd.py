import pickle
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

chunks_path = BASE_DIR / "data" / "chunks.pkl"
db_path = BASE_DIR / "chroma_db"

def load_chunks(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def create_vector_db():
    print(f"Membaca chunks dari: {chunks_path}")

    if not chunks_path.exists():
        print("[ERROR] File chunks.pkl tidak ditemukan. Jalankan igs.py terlebih dahulu.")
        return

    chunks = load_chunks(chunks_path)
    print(f"Total chunks dimuat: {len(chunks)}")

    print("Memuat model embedding...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"  # ganti model embedding jika ram < 8gb
    )

    client = chromadb.PersistentClient(path=str(db_path))

    existing = [c.name for c in client.list_collections()]
    if "ai_tutor_collection" in existing:
        print("Koleksi lama ditemukan, menghapus untuk rebuild bersih...")
        client.delete_collection("ai_tutor_collection")

    print("Membuat koleksi database baru...")
    collection = client.get_or_create_collection(
        name="ai_tutor_collection",
        embedding_function=sentence_transformer_ef
    )

    print("Memasukkan data ke ChromaDB...")
    documents = [item["text"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"  Batch {i//batch_size + 1} selesai ({min(i+batch_size, len(documents))}/{len(documents)})")

    print(f"\nSelesai. Sistem menyimpan {len(chunks)} vektor ke ChromaDB.")

if __name__ == "__main__":
    create_vector_db()