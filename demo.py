from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_huggingface.chat_models import ChatHuggingFace


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# chat_model = ChatHuggingFace(llm=llm)

# print(chat_model.model_id)

tools = [TavilySearchResults(max_results=5)]

prompt = hub.pull("hwchase17/react")

output_parser = ReActSingleInputOutputParser()

agent = create_react_agent(
    llm=llm, tools=tools, prompt=prompt, output_parser=output_parser)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {"input": "Which Secondary School did Ikechukwu Samuel Agbarakwe attend?"})
