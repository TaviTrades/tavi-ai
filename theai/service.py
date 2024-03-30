# from langchain.schema import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI, OpenAI
# from langchain.chains import ConversationChain, LLMChain
# from langchain.memory import (
#     ConversationSummaryMemory,
#     ConversationBufferMemory,
#     ConversationSummaryBufferMemory
# )
# from langchain.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
#     SystemMessagePromptTemplate
# )
# from langchain.agents import AgentType, initialize_agent, Tool
# from langchain_community.utilities import SerpAPIWrapper

# # Load environment variables
# # dotenv.load_dotenv()

# # Initialize the language model
# chat = ChatOpenAI(model="gpt-4")
# llm = OpenAI(temperature=0)

# # Initialize SerpAPIWrapper for search functionality
# search = SerpAPIWrapper()

# def formatted_search(query):
#     results = search.run(query)
#     return f"Search Results: {results}"

# # Setup the tools to be used by the agent
# tools = [
#     Tool(
#         name="Search",
#         func=formatted_search,
#         description="Performs online searches and returns formatted results."
#     ),
# ]

# # Setup the prompt template for the chatbot
# prompt = ChatPromptTemplate(
#     messages=[
#         SystemMessagePromptTemplate.from_template("You are a nice chatbot having a conversation with a human."),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}"),
#     ]
# )

# # Setup memory to maintain conversation context
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Initialize the agent with the tools and language model
# agent_chain = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
#     memory=memory,
# )

# def get_ai_response(input_question):
#     """
#     Function to get the AI response for a given input question.
#     """
#     response = agent_chain.run(input_question)
#     return response


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
from langchain_openai import OpenAI

# Initialize the LLM with OpenAI
llm = OpenAI(temperature=0)

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
            "You are a knowledgeable chatbot skilled in evaluating relevance to topics "
            "like finance, business, biography, history, politics, and notable people. "
            "Rank pleasantries and general greetings with a score of 10. "
            "For each question, provide a relevance score between 1 and 10, where 10 is the most relevant. "
            "Your response should contain only the rank score in number form."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
        SystemMessagePromptTemplate.from_template(
            "Based on the relevance, provide the rank score as a single number."
        )
    ]
)

prompt2 = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a knowledgeable chatbot that can detect when a user has enter a command statement to do something with a financial asset "
            "like open the eurusd chart or show me the eurusd chart or just eurusd. "
            "You just respond with the abbreviation of the asset in the user input in this format cmd:abbreviation. "
            "If the user enters an input that's not a command to open or view a financial asset return 'false'."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
        SystemMessagePromptTemplate.from_template(
            "Based on the command detection, provide the response as 'cmd:abbreviation' or 'false'."
        )
    ]
)


# Define the memory to keep track of the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize LLMChain with the new prompt and memory
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

llm_chain2 = LLMChain(
    llm=llm,
    prompt=prompt2,
    verbose=True,
    memory=memory,
)

# Initialize the agent chain with the tools, LLM, and prompt
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)

def get_ai_response(input_question):
    """
    Function to get the AI response for a given input question.
    """
    llmresponse = llm_chain.run({"question": input_question})
    process_cmd = llm_chain2.run({"question": input_question})
    print(process_cmd)

    if "false" not in process_cmd.lower() and int(llmresponse) >= 5:
        response = agent_chain.run(input_question) + "\n" + process_cmd
    elif "false" in process_cmd.lower() and int(llmresponse) >= 5:
        response = agent_chain.run(input_question)
    elif "false" not in process_cmd.lower():
        response = process_cmd
    else:
        response = "Question not relevant enough for a detailed response."
    return response
