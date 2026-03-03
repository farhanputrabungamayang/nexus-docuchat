import streamlit as st
import PyPDF2
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
import time

# --- 1. KONFIGURASI HALAMAN & LUXURY UI ---
st.set_page_config(page_title="Nexus DocuChat AI", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
        .gradient-text {
            background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 800; font-size: 2.8rem; margin-bottom: 5px; letter-spacing: -1px;
        }
        .subtitle { color: #888; font-size: 1.1rem; margin-bottom: 25px; font-weight: 400; }
        .stButton>button { border-radius: 12px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid rgba(255, 75, 43, 0.2); }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(255, 75, 43, 0.15); }
        [data-testid="stChatMessage"] { border-radius: 16px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid rgba(255,255,255,0.1); background: linear-gradient(145deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); backdrop-filter: blur(10px); }
        .info-box { background-color: rgba(255, 75, 43, 0.05); border-left: 4px solid #FF4B2B; padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SETTING API KEY ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key Google tidak ditemukan di secrets.toml")

# --- 3. STATE MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "doc_processed" not in st.session_state: st.session_state.doc_processed = False
if "doc_stats" not in st.session_state: st.session_state.doc_stats = {"pages": 0, "chunks": 0}

# --- 4. FUNGSI RAG OMNI-READER (PDF, DOCX, TXT) ---
def get_document_text_and_stats(uploaded_files):
    text = ""
    total_docs_or_pages = 0
    for file in uploaded_files:
        if file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            total_docs_or_pages += len(pdf_reader.pages)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted
        elif file.name.endswith('.docx'):
            doc = docx.Document(file)
            total_docs_or_pages += 1
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.name.endswith('.txt'):
            text += file.getvalue().decode("utf-8") + "\n"
            total_docs_or_pages += 1
    return text, total_docs_or_pages

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Anda adalah Asisten AI Nexus yang sangat cerdas.
        Jawab HANYA berdasarkan [Konteks Dokumen] ini. Jika tidak ada, bilang "Maaf, informasi tidak ditemukan di dokumen."
        
        [Konteks Dokumen]:\n{context}\n
        [Pertanyaan User]:\n{user_question}\n
        Jawaban:"""
        
        try:
            model_hidup = next(m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods)
            model = genai.GenerativeModel(model_hidup)
            return model.generate_content(prompt).text
        except Exception as e: return f"⚠️ Gangguan AI: {e}"
    else: return "⚠️ Dokumen belum diproses."

# --- 5. TAMPILAN UI UTAMA ---
st.markdown('<p class="gradient-text">📚 Nexus DocuChat AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Platform RAG Enterprise: Baca PDF, Word, dan TXT dalam Hitungan Detik</p>', unsafe_allow_html=True)

# --- 6. SIDEBAR LUXURY & EXPORT FITUR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135692.png", width=70)
    st.markdown("### 📁 Data Ingestion")
    # SEKARANG BISA BANYAK FORMAT!
    uploaded_files = st.file_uploader("Unggah Dokumen (PDF, DOCX, TXT)", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if st.button("🚀 Ekstrak & Proses Dokumen", use_container_width=True, type="primary"):
        if uploaded_files:
            with st.spinner("Mengekstraksi teks dan membangun Vector Database (Lokal)..."):
                raw_text, total_count = get_document_text_and_stats(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                
                st.session_state.doc_processed = True
                st.session_state.doc_stats["pages"] = total_count
                st.session_state.doc_stats["chunks"] = len(text_chunks)
                
                st.toast(f"Berhasil membaca {total_count} unit dokumen!", icon="✅")
                time.sleep(1)
                st.toast("Vector Database berhasil dibangun!", icon="🧠")
        else:
            st.error("⚠️ Harap unggah minimal 1 dokumen!")

    if st.session_state.doc_processed:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Status Dokumen")
        with st.container(border=True):
            st.write(f"📄 **Unit Terbaca:** {st.session_state.doc_stats['pages']}")
            st.write(f"🧩 **Potongan Data:** {st.session_state.doc_stats['chunks']} chunks")
            st.write("🟢 **Status:** Siap Berdiskusi")
            
        if st.button("🗑️ Bersihkan Obrolan", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        # FITUR BARU: DOWNLOAD CHAT HISTORY!
        if st.session_state.messages:
            st.markdown("---")
            st.markdown("### 💾 Export Data")
            chat_str = "=== RIWAYAT DISKUSI NEXUS AI ===\n\n"
            for msg in st.session_state.messages:
                role = "USER" if msg["role"] == "user" else "NEXUS AI"
                chat_str += f"[{role}]:\n{msg['content']}\n\n{'-'*50}\n\n"
            
            st.download_button(
                label="📥 Download Hasil Diskusi (.txt)",
                data=chat_str,
                file_name="Nexus_AI_Chat_History.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.markdown("---")
    st.caption("🔒 Privacy First: Local Embedding via HuggingFace.")

# --- 7. MAIN CHAT AREA ---
if not st.session_state.messages and not st.session_state.doc_processed:
    st.markdown("""
        <div class="info-box">
            <h4>👋 Selamat datang di Nexus DocuChat!</h4>
            <p>Aplikasi ini memungkinkan Anda untuk mengobrol dengan dokumen Anda sendiri (PDF, Word, TXT) menggunakan kecerdasan buatan.</p>
        </div>
    """, unsafe_allow_html=True)

elif not st.session_state.messages and st.session_state.doc_processed:
    st.info("✨ Dokumen Anda sudah siap! Coba tanyakan sesuatu, misalnya: *'Tolong buatkan ringkasan dari dokumen ini.'*")

for message in st.session_state.messages:
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan sesuatu ke Nexus AI tentang dokumen Anda..."):
    if not st.session_state.doc_processed:
        st.warning("⚠️ Proses dokumen Anda terlebih dahulu di menu samping.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"): st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Berpikir..."):
                reply = user_input(prompt)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})