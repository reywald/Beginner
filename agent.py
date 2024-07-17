from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use WebBaseLoader to extract data from the dasa.org website
web_loader = WebBaseLoader("https://www.dasa.org")
web_contents = web_loader.load()

# Use RecursivCharacterTextSplitter to split text into document fragments
text_splitter = RecursiveCharacterTextSplitter()
document_fragments = text_splitter.split_documents(documents=web_contents)

# Get vector embedding function/model
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
)

# Create Vector Store, initialized from documents and embedding algorithm
vector_store = FAISS.from_documents(
    documents=document_fragments, embedding=embeddings)
retriever = vector_store.as_retriever()
