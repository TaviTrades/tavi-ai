from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from .agentQFlow import get_agentQFlow_response
from .knowledgeQFlow import get_knowledgeQFlow_response

# Define the examples for training the model

model = ChatOpenAI(model="gpt-4")



prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    ("user", "{input}")
])

chain = prompt | model


examples = [
    {"input": "I'm sorry, as an AI, I don't have real-time capabilities to provide current information or updates. I recommend checking a reliable financial news source or a cryptocurrency exchange for the most up-to-date Bitcoin prices", "output": "NI"},
    {"input": "I'm sorry for any misunderstanding, but as a text-based AI model developed by OpenAI, I don't have the functionality to open charts or perform any live actions on the internet", "output": "NI"}
]


to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)


example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "what is the price of BTC?"})







# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)




final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a AI that returns NI or NNI, NI if the {input} is a question that needs internet access or external accesss to answer or perform a task and NNI if {input} can answer the question or input without internet access"),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# Define the chain
section_chain = final_prompt | model

def get_qFlow_response(input_question):
    """
    Function to get the AI response for a given input question.
    """
    question_response = chain.invoke({"input": input_question})
    print('This is the response' + question_response.content)


    conditional_response = section_chain.invoke({"input": question_response.content})
    conditional_response__content = conditional_response.content 

    if conditional_response__content == 'NI':
        response = get_agentQFlow_response(input_question)
    elif conditional_response__content == 'NNI':
        response = get_knowledgeQFlow_response(input_question)
    else:
        response = "Invalid check value. Please set check to either 'NI' or 'NNI'."
    
    return response