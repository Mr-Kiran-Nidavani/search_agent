from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool

# Create a DuckDuckGo search tool instance
duckduckgo_search = DuckDuckGoSearchRun()
search_tool = Tool(name="DuckDuckGo_Search", func=duckduckgo_search.run, description="search the web for required information for user query")

wikipedia_wrap = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrap)



def create_search_paper(data: str, fileName: str = "research_output.txt"):
    with open(fileName, "a", encoding="utf-8") as file:
        file.write(f"{data}\n")

    print(f"File successfully created: {fileName}")


save_tool = Tool(name="Save_Search_Paper", func=create_search_paper, description="Save structured reaserch data to a file")