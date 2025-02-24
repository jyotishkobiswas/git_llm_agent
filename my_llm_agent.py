#Codebase from github location https://github.com/Rachnog/intro_to_llm_agents/blob/main/4_agents.ipynb
#
#Additional work done by Jyotishko

#Simple LLM and RAG based agent for QUestion answering

import os
import yaml

tyly_api_key = os.getenv("TAVILY_API_KEY")

lsmith_api_key = os.getenv("LSMITH_API_KEY")

openai_api_key = os.getenv("OPENAI_API_KEY")

#from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
#from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.tools import tool

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

@tool
def calculate_length_tool(a: str) -> int:
    """The function calculates the length of the input string."""
    return len(a)

@tool
def calculate_uppercase_tool(a: str) -> int:
    """The function calculates the number of uppercase characters in the input string."""
    return sum(1 for c in a if c.isupper())

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# Tool webpage information and stored it in FAISS vector store

loader = WebBaseLoader("https://awards.3ai.in/acme-2024-winners/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "Jyotishko_Biswas_search",
    "Search for information about Jyotishko Biswas. For any questions about Jyotishko Biswas, you must use this tool!",
)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

message_history = ChatMessageHistory()

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt = hub.pull("wfh/langsmith-agent-prompt:5d466cbc")

prompt

tools = [retriever_tool, tavily_tool, calculate_length_tool, calculate_uppercase_tool]
for t in tools:
    print(t.name, t.description)

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_chat_history.invoke(
    {
        "input": "Identify if Dr. Santosh Karthikeyan Viswanatha won any award in 2024. Also let me know the count of popel who won the same award",
       
    }, 
    config={"configurable": {"session_id": "<foo>"}}
)


from langchain.output_parsers import PydanticOutputParser
#from langchain_core.pydantic_v1 import Field, validator
from pydantic import BaseModel, Field, validator

# Define your desired data structure.
class DesiredStructure(BaseModel):
    question: str = Field(description="the question asked")
    numerical_answer: int = Field(description="the number extracted from the answer, text excluded")
    text_answer: str = Field(description="the text part of the answer, numbers excluded")
parser = PydanticOutputParser(pydantic_object=DesiredStructure)

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_and_model = prompt | model
output = prompt_and_model.invoke({
    "query": "Identify if Jyotishko Biswas won any award in 2024. Also let me know the count of popel who won the same award"}
)

output
