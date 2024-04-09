import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate


model = ChatOpenAI(model="gpt-4")


cmdexamples = [
   {"input": "open eurusd chart", "output": "asset=EURUSD"},
   {"input": "view the chat in as a line chart ", "output": "chartType=line_chart"},
   {"input": "set the time frame to 15 minutes", "output": "timeFrame=15m"},
   {"input":"use MACD indicator", "output":"indicator=[MACD]"},
   {"input":"add the MACD and stocastic indicator", "output":"indicator=[MACD, stocastic]"},
   {"input":"show the AUDUSD chart and add the MACD and stocastic indicator also set the timeframe to 30 mins", "output":"asset=eurusd, timeframe=30m, indicator=[MACD, stocastic]"}
   # {"input": "2+2", "output": "4"},

]

to_vectorize = [" ".join(example.values()) for example in cmdexamples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=cmdexamples)

example_selector = SemanticSimilarityExampleSelector(
  vectorstore=vectorstore,
  k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
# example_selector.select_examples({"input": "open the eurusd chart?"})


# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)


# print(few_shot_prompt.format(input="Open the appl chart"))


final_prompt = ChatPromptTemplate.from_messages(
  [
      ("system", "You are an AI that response doesn't open the chart but just responds with the currency pair or asset mentioned and other things related to chart also follow the format in the prompt"),
      few_shot_prompt,
      ("human", "{input}"),
  ]
)

cmdchain = final_prompt | model

def get_cmdflow_response(input_question):
    
    """
    Function to get the AI response for a given input question.
    """
    # llmresponse = llm_chain.run({"question": input_question})

    response = cmdchain.invoke(input_question)
    response = "cmdflow: " + response.content
    return response