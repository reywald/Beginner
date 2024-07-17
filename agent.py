from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use WebBaseLoader to extract data from the dasa.org website
web_loader = WebBaseLoader("https://www.dasa.org")
web_contents = web_loader.load()

# Use RecursivCharacterTextSplitter to split text into document fragments
text_splitter = RecursiveCharacterTextSplitter()
document_fragments = text_splitter.split_documents(documents=web_contents)
