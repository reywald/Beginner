from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv


# Load the HuggingFace API Token environment variable
load_dotenv()

# Use WebBaseLoader to extract data from website
web_loader = WebBaseLoader("https://www.dasa.org")
web_text = web_loader.load()
