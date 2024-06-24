from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load the HuggingFace API Key environment variable
load_dotenv()

# Create an LLM instance using the HuggingFaceEndpoint integration
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# Create prompt template consisting of a system and human message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Physics teacher."),
    ("human", "{input}")
])

output_parser = StrOutputParser()

# Create a pipeline consisting of the llm and prompt
chain = prompt | llm | output_parser

print(chain.invoke("What is the first law of Thermodynamics?"))
