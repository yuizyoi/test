import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set up Streamlit UI
st.title("AI Chatbot with RAG")

# Sidebar: API Key and Temperature
with st.sidebar:
    st.title('Google API Key Settings')
    st.write("Get Google API Key [here](https://makersuite.google.com/app/apikey)")
    google_api_key = st.text_input('Enter your Google API key:', type='password')
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# Upload PDF file
st.subheader("Upload a PDF to Enable RAG-based Q&A")

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    save_dir = "uploaded_files"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
pdf_path = save_uploaded_file(uploaded_file) if uploaded_file else None

#pdf_path = "sample.pdf"
# Load PDF, Split into Chunks, and Store in Vector DB
def load_pdf_data(file, chunk_size=500, chunk_overlap=50):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Create FAISS Vector Store
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

# Setup QA System
def setup_qa_system(vector_store, temperature):
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=temperature, google_api_key=google_api_key)
    prompt_template = """คุณเป็นผู้ช่วยที่เชี่ยวชาญในการตอบคำถามจากเอกสาร
    
    โปรดใช้ข้อมูลต่อไปนี้เพื่อตอบคำถาม:
    {context}
    
    คำถาม: {question}
    
    โปรดตอบโดยละเอียดและชัดเจน พร้อมอ้างอิงข้อมูลจากเอกสาร:"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type="stuff",  # หรือใช้ "map_reduce" สำหรับเอกสารยาวๆ
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            ),
        }
    )
    return qa_chain

    # retriever = vector_store.as_retriever()
    # return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Processing PDF file
if google_api_key and pdf_path:
    with st.spinner("Processing PDF..."):
        
        #Load PDF and create vector store
        chunks = load_pdf_data(pdf_path)
        vector_store = create_vector_store(chunks)
        
        # Setup QA system
        qa_chain = setup_qa_system(vector_store, temperature)
        st.success("PDF successfully loaded into vector database!")

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask me a question?"):
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response from LLM
        with st.chat_message("ai"):
            with st.spinner("Retrieving and answering..."):
                result = qa_chain({"query": user_input})
                response = result["result"]
                st.write(response)
                st.session_state["chat_history"].append({"role": "ai", "content": response})
