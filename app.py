import streamlit as st
import os
import random
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- 1. 基礎設定 ---
load_dotenv()

st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️", layout="wide") # layout="wide" 讓畫面寬一點
st.title("⚖️ 公司勞基法智慧問答")
st.caption("我是你的 AI 法務助理，關於請假、加班費、資遣費的問題都可以問我！")

# 讀取 Key
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not groq_api_key or not hf_token:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        hf_token = st.secrets["HF_TOKEN"]
    except FileNotFoundError:
        pass

if not groq_api_key or not hf_token:
    st.error("❌ 錯誤：找不到 API Key！請確認 .env 檔案或 Secrets 設定。")
    st.stop()

# --- 2. 初始化資源 ---
@st.cache_resource
def load_resources():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever()
    except Exception as e:
        return None, None, f"載入資料庫失敗: {e}"
    
    try:
        # 使用 Llama 3.3
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0.3
        )
    except Exception as e:
        return None, None, f"載入 LLM 失敗: {e}"

    return retriever, llm, None

retriever, llm, error_msg = load_resources()

if error_msg:
    st.error(error_msg)
    st.stop()

# --- ★★★ 新增功能區塊：側邊欄 (Sidebar) ★★★ ---
with st.sidebar:
    st.header("🔧 設定與工具")

    # 功能 3: 角色切換 (Prompt Engineering)
    st.subheader("1️⃣ 設定你的身分")
    role = st.radio(
        "AI 將根據身分調整回答語氣：",
        ["勞工 (想了解自身權益)", "雇主 (想合規管理風險)"],
        index=0
    )

    st.markdown("---")

    # 功能 2: 加班費試算器
    st.subheader("2️⃣ 簡易加班費試算")
    salary = st.number_input("月薪 (元)", value=30000, step=1000)
    hours_1 = st.number_input("平日加班 (前2小時內)", value=0, step=1)
    hours_2 = st.number_input("平日加班 (第3小時起)", value=0, step=1)
    
    # 簡單計算：時薪 = 月薪 / 240, 前2小時*1.34, 後2小時*1.67
    hourly_rate = salary / 240
    overtime_pay = (hours_1 * hourly_rate * 1.34) + (hours_2 * hourly_rate * 1.67)
    
    if st.button("計算預估加班費"):
        st.success(f"💰 預估加班費：**{int(overtime_pay)}** 元")
        st.caption("註：此為概算 (1.34/1.67倍)，實際依公司規定為準。")

    st.markdown("---")

    # 功能 1: 清除對話紀錄
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 3. 設定 RAG Chain (根據側邊欄的角色動態調整) ---

if "勞工" in role:
    system_prompt_content = (
        "你是一位站在勞工立場的法律顧問。"
        "回答時請著重於如何保護勞工權益、如何計算應得薪資，"
        "若公司可能違法，請溫和提示可向勞工局申訴的管道，或是需要蒐集什麼證據。"
        "回答語氣要同理且堅定。"
        "請根據以下提供的法規條文內容來回答使用者的問題。"
        "\n\n{context}"
    )
else:
    system_prompt_content = (
        "你是一位企業法務顧問。"
        "回答時請著重於公司如何遵守勞基法以避免罰則與勞資糾紛。"
        "請提供合規的建議與管理措施，提醒雇主應注意的法規細節。"
        "回答語氣要專業、客觀且謹慎。"
        "請根據以下提供的法規條文內容來回答使用者的問題。"
        "\n\n{context}"
    )

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_content), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 4. 聊天歷史紀錄 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. 處理輸入邏輯 (包含功能 5: 滿意度回饋) ---
def process_question(user_question):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("查詢法條中..."):
            try:
                response = rag_chain.invoke({"input": user_question})
                answer = response["answer"]
                st.markdown(answer)
                
                with st.expander("查看參考法條來源"):
                    if "context" in response and response["context"]:
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**來源 {i+1}:** {doc.page_content[:100]}...")
                    else:
                        st.markdown("無具體參考來源。")
                
                # 功能 5: 滿意度回饋按鈕
                # 使用 unique key 避免重複
                btn_key_seed = len(st.session_state.messages)
                col_up, col_down, col_space = st.columns([1, 1, 8])
                with col_up:
                    if st.button("👍", key=f"like_{btn_key_seed}"):
                        st.toast("感謝您的回饋！(已記錄)")
                with col_down:
                    if st.button("👎", key=f"dislike_{btn_key_seed}"):
                        st.toast("我們會再改進！")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"發生錯誤：{e}")

# --- 6. 動態題庫邏輯 ---
ALL_QUESTIONS = [
    "加班費怎麼算？",
    "特休假沒休完可以換錢嗎？",
    "被資遣了，資遣費怎麼算？",
    "颱風假沒去上班會扣薪水嗎？",
    "試用期有薪水嗎？",
    "離職需要幾天前預告？",
    "國定假日上班薪水怎麼算？",
    "請病假會扣全薪嗎？",
    "老闆不給加班費怎麼辦？",
    "一天工時上限是多少？",
    "婚假有幾天？",
    "產假薪水怎麼算？"
]

if "current_options" not in st.session_state:
    shuffled_questions = ALL_QUESTIONS.copy()
    random.shuffle(shuffled_questions)
    st.session_state.current_options = shuffled_questions[:4]
    st.session_state.reserve_questions = shuffled_questions[4:]

def replace_question(index):
    if len(st.session_state.reserve_questions) > 0:
        new_q = st.session_state.reserve_questions.pop(0)
        old_q = st.session_state.current_options[index]
        st.session_state.reserve_questions.append(old_q)
        st.session_state.current_options[index] = new_q

# --- 7. 介面互動區 ---

st.write("💡 **快速提問 (點擊後自動換題)：**")
cols = st.columns(4)
trigger_question = None

for i, col in enumerate(cols):
    question_text = st.session_state.current_options[i]
    if col.button(question_text, key=f"btn_{i}"):
        trigger_question = question_text
        replace_question(i)

chat_input = st.chat_input("請輸入您的勞基法問題...")

if trigger_question:
    process_question(trigger_question)
    st.rerun()
elif chat_input:
    process_question(chat_input)