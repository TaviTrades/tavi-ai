
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from .qFlow import get_qFlow_response
from .cmdflow import get_cmdflow_response


# Define the examples for training the model

model = ChatOpenAI(model="gpt-4")


examples = [
    {"input": "What is the price of BTC?", "output": "Q"},
    {"input": "Who is the CEO of apple?", "output": "Q"},
    {"input": "What is the current price of apple stock?", "output": "Q"},
    {"input": "Open the EURUSD chart", "output":"cmd"},
    {"input": "view chart as line chart", "output":"cmd"},
    {"input": "Add a bollinger band to the chart", "output":"cmd"},
    {"input": "Add a stocastic indicator to the chart", "output":"cmd"},
]

# Check that all input and output values are strings
for example in examples:
    assert isinstance(example["input"], str), f"Invalid input: {example['input']}"
    assert isinstance(example["output"], str), f"Invalid output: {example['output']}"

# Vectorize the examples
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# Define the example selector
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# Define the few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

# Define the final prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a AI that returns Q or cmd, Q if the {input} is more of a question than a task to be done on the trading terminal and cmd if it's more of a command or a task to be done on the trading terminal"),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# Define the chain
chain = final_prompt | model

def get_qcCheck_response(input_question):
    """
    Function to get the AI response for a given input question.
    """
    
    response = chain.invoke({"input": input_question})
    response_content = response.content 

    if response_content == 'Q':
        response = get_qFlow_response(input_question)
    elif response_content == 'cmd':
        response = get_cmdflow_response(input_question)
    else:
        response = "Invalid check value. Please set check to either 'q' or 'cmd'."
    
    return response





