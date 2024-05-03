from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from .qcCheck import get_qcCheck_response


model = ChatOpenAI(model="gpt-4")

vectorstore = DocArrayInMemorySearch.from_texts(
    ["bears like to eat honey"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

memory = ConversationBufferMemory(return_messages=True)

memory.load_memory_variables({})

{'history': []}

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)



def get_memory_response(input_question,relation_status):
    """
    Function to get the AI response for a given input question.
    """
    
    response = chain.invoke({"input": input_question})
    response_content = response.content 

    if relation_status == 'F':
        response = get_qcCheck_response(input_question)
    elif relation_status == 'NF':
        response = "This question isn't related to finance, history, trading or business"
    else:
        response = "Invalid check value."

    memory.save_context(input_question, {"output": response})
    print(memory.load_memory_variables({}))
    return response
