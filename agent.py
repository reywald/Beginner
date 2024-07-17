from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# Load the HuggingFace API Token and the Tavily API Key
load_dotenv()

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

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="DASA_search",
    description="Search for information about DASA. For any questions about DASA, you must use this tool"
)

# Create tool used for searching the internet
search_tool = TavilySearchResults()

# Create list of prepared AI tools
tools = [retriever_tool, search_tool]