from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


## LLM
'''
def get_llm():
    #model_id = "mistralai/mixtral-8x7b-instruct-v01"
    model_id = "ibm/granite-13b-instruct-v2"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm
'''
SUPPORTED_MODELS = [
    "google/flan-t5-xl",
    "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-13b-chat",
    "mistralai/mistral-medium-2505",
]

def get_llm():
    for model_id in SUPPORTED_MODELS:
        try:
            parameters = {
                GenParams.MAX_NEW_TOKENS: 256,
                GenParams.TEMPERATURE: 0.5,
            }
            project_id = "skills-network"
            return WatsonxLLM(
                model_id=model_id,
                url="https://us-south.ml.cloud.ibm.com",
                project_id=project_id,
                params=parameters,
            )
        except Exception as e:
            print(f"⚠️ Model {model_id} failed: {e}")
    raise RuntimeError("❌ No supported LLMs available in this environment.")

## Document loader
def document_loader(file):
    if file is None:
        return []  # no file provided
    #loader = PyPDFLoader(file.name)
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document


## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding


## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


## Retriever
def retriever(file):
    splits = document_loader(file)
    if not splits:  # handle no file case
        return None
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever


## QA Chain
def retriever_qa(file, query):
    if file is None:
        return "⚠️ Please upload a PDF file first."

    llm = get_llm()
    retriever_obj = retriever(file)
    if retriever_obj is None:
        return "⚠️ Could not process the PDF file."

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True,
    )
    response = qa.invoke({"query": query})
    return response.get("result", "⚠️ No answer generated.")


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        #gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="file"),
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="QA Bot with LangChain + Watsonx",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
)


# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port=7860)
