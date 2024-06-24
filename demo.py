from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the HuggingFace API Token environment variable
load_dotenv()

# Use WebBaseLoader to extract data from website
web_loader = WebBaseLoader("https://www.dasa.org")
web_text = web_loader.load()

# Use RecursiveCharacterTextSplitter to split text into fragments
text_splitter = RecursiveCharacterTextSplitter()
text_fragments = text_splitter.split_documents(documents=web_text)

# Get vector embedding function/model
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
)

# Create VectorStore, initialized from document fragments and embedding function
vector_store = FAISS.from_documents(
    documents=text_fragments, embedding=embeddings)

# Create a prompt template to contain the retrieved data and user input
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Create an LLM using HuggingFaceHub API token
llm = HuggingFaceEndpoint(
    # repo_id="microsoft/Phi-3-mini-4k-instruct",
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# print(prompt.invoke(
#     {"input": "What are the talent products delivered by DASA?", "context": "Hello"}))

# Create a document chain to send prompt to LLM
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Create a retriever to retrieve data and a retriever chain
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Invoke the chain
response = retrieval_chain.invoke(
    {"input": "What are the talent products delivered by DASA?"})
print(f"{response['answer'] = }")
