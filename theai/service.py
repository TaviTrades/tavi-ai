
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from .memory import get_memory_response


# Define the examples for training the model

model = ChatOpenAI(model="gpt-4")


examples = [
    {"input": "What is the price of BTC?", "output": "F"},
    {"input": "Who is the CEO of apple?", "output": "F"},
    {"input": "What is the current price of apple stock?", "output": "F"},
    {"input": "Open the EURUSD chart", "output":"F"},
    {"input": "view chart as line chart", "output":"F"},
    {"input": "Add a bollinger band to the chart", "output":"F"},
    {"input": "How can i bake cake", "output":"NF"},
    {"input": "What cloth was kim kadasian wearing at the fashion show?", "output":"NF"},
    {"input": "What is meteorology?", "output":"NF"},
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
        ("system", "You're a AI that returns F or NF, F if the {input} is related to finance, history, business, companies, coporate world, trading, stocks, forex, crypto e.t.c  and NF if it's not related to any of that."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# Define the chain
chain = final_prompt | model

def get_ai_response(input_question):
    """
    Function to get the AI response for a given input question.
    """
    
    response = chain.invoke({"input": input_question})
    response_content = response.content 

    if response_content == 'F':
        response = get_memory_response(input_question, 'F')
    elif response_content == 'NF':
        response = get_memory_response(input_question, 'NF')
    else:
        response = "Invalid check value. Please set check to either 'related to finance' or 'note'."
    
    return response





