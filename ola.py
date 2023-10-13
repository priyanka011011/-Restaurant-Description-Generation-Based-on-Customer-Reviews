import langchain
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gradio as gr

def init_llm():
    llm = Ollama(
        model="llama2:7b",
        temperature=0.3,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm

template = """You are a ai specialised in summarizing food reviews. You always summarize the reviews to 100 words.
Review: {question}
"""

llm = init_llm()

def get_answer_interface(query):
    prompt = template.format(question=query)
    res = llm(prompt)
    print(res) 
    return res  


iface = gr.Interface(
    fn=get_answer_interface, 
    inputs="text", 
    outputs="text", 
    live=False
)
iface.launch()
