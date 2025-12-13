from langgraph.graph import StateGraph, START, END
#import google chat model
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# create a state

class LLMState(TypedDict):
    
    question: str
    answer: str
    
def llm_qa(state: LLMState) -> LLMState:
    
    question = state['question']
    
    prompt = f'Answer the following question {question}'

    answer = model.invoke(prompt).content

    state['answer'] = answer

    return state


# create graph

graph = StateGraph(LLMState)

# add a node
graph.add_node("llm_qa", llm_qa)

# add edges
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

# run the graph
workflow = graph.compile()

# excute

intial_state = {'question': 'What is the capital of France?'}

final_state = workflow.invoke(intial_state)

print(final_state)