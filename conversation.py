from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Extract data from website
web_loader = WebBaseLoader("https://www.dasa.org")
web_text = web_loader.load()

# Split text into fragments
text_splitter = RecursiveCharacterTextSplitter()
web_documents = text_splitter.split_documents(web_text)

# Get Embedding model/algorithms/function
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

# Instantiate Vector Store and its Retriever
vector_store = FAISS.from_documents(
    documents=web_documents, embedding=embeddings)
retriever = vector_store.as_retriever()

# Import HuggingFaceEndpoint and create an LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
