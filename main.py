from urllib import response
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResponseStructure(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

chat = ChatOpenAI(
    model="gpt-4o-mini",  # Or other models supported by OpenRouter
    temperature=0.7,
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)

parser = PydanticOutputParser(pydantic_object=ResponseStructure)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You search expert, who will help to generate search queries
     Answers the user query using neccessory tool and wrap the output in provided format and no extra text \n {response_structure}.
     """),
    ("placeholder", "{chat_history}"),
    ("user", "{user_query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(response_structure=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("Enter your query: ")
raw_response = agent_executor.invoke({"user_query": query})

print(raw_response)
structured_response = parser.parse(raw_response.get("output"))

print(structured_response)