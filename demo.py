from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


# Load the HuggingFace API Token environment variable
load_dotenv()

# Use WebBaseLoader to extract data from website
web_loader = WebBaseLoader("https://www.dasa.org")
web_text = web_loader.load()

# Use RecursiveCharacterTextSplitter to split text into fragments
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(documents=web_text)

# Convert fragmented text into vector embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)