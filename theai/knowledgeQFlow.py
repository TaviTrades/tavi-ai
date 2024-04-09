import os
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI, ChatOpenAI




# Initialize the LLM with OpenAI
# llm = OpenAI(temperature=0)
llm = ChatOpenAI(model="gpt-4")


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

# Initialize LLMChain with the new prompt and memory
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)


def get_knowledgeQFlow_response(input_question):
    
    """
    Function to get the AI response for a given input question.
    """
    # llmresponse = llm_chain.run({"question": input_question})

    response = llm_chain.run(input_question)
    response = "knowledge: " + response
    return response