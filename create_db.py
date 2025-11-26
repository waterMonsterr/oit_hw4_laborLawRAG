import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# 1. 載入環境變數
load_dotenv()

# 設定資料夾路徑
DATA_PATH = "labor_law.txt"
DB_PATH = "vectorstore"  # 資料庫要存的地方

def create_vector_db():
    print("🚀 開始建立向量資料庫...")

    # 2. 讀取勞基法文件
    if not os.path.exists(DATA_PATH):
        print(f"錯誤：找不到 {DATA_PATH}，請確認檔案是否存在。")
        return

    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()
    print(f"📄 讀取文件成功，共 {len(documents)} 份文件")

    # 3. 切分文字 (Chunking)
    # 法規條文比較嚴謹，我們可以切小一點，確保 context 精準
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # 每個區塊 500 字
        chunk_overlap=50   # 重疊 50 字，避免切斷語意
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  文件已切分為 {len(chunks)} 個區塊")

    # 4. 初始化 Embedding 模型 (使用 HuggingFace 的開源模型)
    # 這裡使用支援中文較好的模型，或者通用的 all-MiniLM-L6-v2
    print("🧠 正在下載 Embedding 模型 (第一次會比較久)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. 建立並儲存向量資料庫 (FAISS)
    print("💾 正在建立索引並儲存...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    
    print(f"✅ 完成！向量資料庫已儲存至 '{DB_PATH}' 資料夾")

if __name__ == "__main__":
    create_vector_db()