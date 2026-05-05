import re
import fitz
from tqdm import tqdm
import pickle
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

pdf_folder = BASE_DIR / "data" / "pdf"
output_path = BASE_DIR / "data" / "chunks.pkl"

SUBTEST_KEYWORDS = ["PPU", "PBM", "LBI", "LBE", "PU", "PK", "PM"]

def detect_subtest(filename: str) -> str:
    """
    Deteksi subtest dari nama file.
    Contoh: 'to_PBM_py8.pdf' -> 'PBM'
            'PK_to2.pdf'     -> 'PK'
            'latihan_pm.pdf' -> 'PM'
    """
    name_upper = filename.upper()
    for keyword in SUBTEST_KEYWORDS:
        pattern = rf"(?:^|[_\-\s\d\[\]()]){keyword}(?:[_\-\s\d\[\]()]|$)"
        if re.search(pattern, name_upper):
            return keyword
    return "UNKNOWN"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(str(pdf_path))
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text + "\n"
    return full_text

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def process_pdf_folder(pdf_folder, output_path):
    print(f"Membaca PDF dari lokasi: {pdf_folder}")

    if not pdf_folder.exists():
        print(f"[ERROR] Folder tidak ditemukan: {pdf_folder}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     
        chunk_overlap=75,      
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    subtest_counts = {k: 0 for k in SUBTEST_KEYWORDS}
    subtest_counts["UNKNOWN"] = 0

    pdf_files = [f for f in pdf_folder.iterdir() if f.suffix.lower() == ".pdf"]

    if not pdf_files:
        print("[ERROR] Tidak ada file PDF di folder.")
        return

    for file in tqdm(pdf_files):
        subtest = detect_subtest(file.name)
        print(f"Mengeksekusi: {file.name} -> subtest: {subtest}")

        text = extract_text_from_pdf(file)
        text = clean_text(text)

        if len(text.strip()) < 5: #ganti minimal 20
            print(f"  [SKIP] Teks terlalu pendek, kemungkinan PDF scan.")
            continue

        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            if len(chunk.strip()) > 30:
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": file.name,
                        "subtest": subtest
                    }
                })
                subtest_counts[subtest] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nSelesai. Total {len(all_chunks)} chunks tersimpan.")
    print("\nRincian per subtest:")
    for subtest, count in subtest_counts.items():
        if count > 0:
            print(f"  {subtest}: {count} chunks")

    if subtest_counts["UNKNOWN"] > 0:
        print(f"\n[PERINGATAN] {subtest_counts['UNKNOWN']} chunks tidak terdeteksi subtestnya.")
        print("  Periksa nama file yang menghasilkan UNKNOWN dan sesuaikan.")

if __name__ == "__main__":
    process_pdf_folder(pdf_folder, output_path)