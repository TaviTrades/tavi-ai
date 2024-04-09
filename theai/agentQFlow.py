import os
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import OpenAI, ChatOpenAI




# Initialize the LLM with OpenAI
# llm = OpenAI(temperature=0)
llm = ChatOpenAI(model="gpt-4")

# Define a tool for performing online searches
search = SerpAPIWrapper()

def formatted_search(query):
    results = search.run(query)
    return f"Search Results: {results}"

# Define the tools for the agent
tools = [
    Tool(
        name="Search",
        func=formatted_search,
        description="Performs online searches and returns formatted results."
    ),
]

# Updated prompt with clearer instructions and scoring system
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You're a chatbot that answers a question to the best of your ability."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
        SystemMessagePromptTemplate.from_template(
            "Based on your knowledge, provide the best possible answer."
        )
    ]
)



# Define the memory to keep track of the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



# Initialize the agent chain with the tools, LLM, and prompt
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)

def get_agentQFlow_response(input_question):
    
    """
    Function to get the AI response for a given input question.
    """
    # llmresponse = llm_chain.run({"question": input_question})

    response = agent_chain.run(input_question)
    response = "agentQFlow: " + response
    return response