from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Extract data from website
web_loader = WebBaseLoader("https://www.dasa.org")
web_text = web_loader.load()

# Split text into fragments
text_splitter = RecursiveCharacterTextSplitter()
web_fragments = text_splitter.split_text(web_text)