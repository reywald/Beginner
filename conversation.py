from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import (HuggingFaceEndpoint,
                                   HuggingFaceEndpointEmbeddings)
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

# Create prompt to contain user input, chat history and message
# to generate a search query
prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", """Given the above conversation, generate a 
     search query to look up to get information relevant
     to the conversation""")
])

# Instantiate Retriever chain from llm, vector retriever and prompt
retriever_chain = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=prompt_search_query)

# Create prompt to get response to user input
prompt_get_answer = ChatPromptTemplate.from_messages([
    ("system", """Answer the user's questions based on 
     the below context:\n\n{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

# Send Retrieved documents to LLM and get response
document_chain = create_stuff_documents_chain(
    llm=llm, prompt=prompt_get_answer)
