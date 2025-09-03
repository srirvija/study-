##pip install gradio langchain chromadb pypdf sentence-transformers huggingface_hub
##export HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"
###export OPENAI_API_KEY="sk-xxxx..."
from langchain import HuggingFaceHub
###OpenAI GPT
##from langchain_openai import ChatOpenAI
###Run Local Model
##from transformers import pipeline
##from langchain.llms import HuggingFacePipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
import warnings

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")


# ✅ Hugging Face model for QA
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",   # free model
        model_kwargs={"temperature": 0.5, "max_length": 256},
    )

'''
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=256)
'''

'''
##downloads the model locally
def get_llm():
    pipe = pipeline("text-generation", model="google/flan-t5-base")
    return HuggingFacePipeline(pipeline=pipe)
'''

# ✅ Hugging Face embedding model
def embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ✅ Document loader
def document_loader(file):
    if file is None:
        return []
    try:
        loader = PyPDFLoader(file)   # file is a path string
        return loader.load()
    except Exception as e:
        print(f"⚠️ Could not load document: {e}")
        return []


# ✅ Text splitter
def text_splitter(data):
    if not data:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_documents(data)


# ✅ Vector DB
def vector_database(chunks):
    if not chunks:
        return None
    embeddings = embedding_model()
    return Chroma.from_documents(chunks, embeddings)


# ✅ Retriever
def retriever(file):
    splits = document_loader(file)
    if not splits:
        return None
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    if vectordb is None:
        return None
    return vectordb.as_retriever()


# ✅ QA Chain
def retriever_qa(file, query):
    if file is None:
        return "⚠️ Please upload a PDF file first."

    retriever_obj = retriever(file)
    if retriever_obj is None:
        return "⚠️ Could not process the PDF file."

    try:
        llm = get_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=True,
        )
        response = qa.invoke({"query": query})
        return response.get("result", "⚠️ No answer generated.")
    except Exception as e:
        return f"❌ Error while generating answer: {e}"


# ✅ Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath",
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here...",
        ),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="QA Bot with LangChain + HuggingFace",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
)


# ✅ Launch app
if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)
