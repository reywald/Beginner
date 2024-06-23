from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


# Load the HuggingFace API Token environment variable
load_dotenv()

# Use WebBaseLoader to extract data from website
web_loader = WebBaseLoader("https://www.dasa.org")
web_text = web_loader.load()

# Use RecursiveCharacterTextSplitter to split text into fragments
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(documents=web_text)