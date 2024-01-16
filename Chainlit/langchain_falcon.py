from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory


import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

#repo_id = "tiiuae/falcon-7b-instruct"
#repo_id =  "gpt2-medium"
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.1, "max_new_tokens":700})

template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

# Observe que "chat_history" está presente en la plantilla de la conversación
template = """Eres un chatbot amigable llamado InnovaBot que mantiene una conversación con un humano. Por favor no inventar más preguntas.

Conversación anterior:
{chat_history}

Nueva pregunta del humano: {question}
Respuesta:"""


@cl.on_chat_start
async def main():

    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="image.png")
    ]
    await cl.Message(content="Bienvenido, Soy InnovaBot. ¿Cómo puedo ayudarte?", elements=elements).send()

    # Instantiate the chain for that user session
    prompt = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(memory_key="chat_history")
    #prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose = True, memory=memory)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    
    # Call the chain asynchronously
    res = await llm_chain.apredict(question=message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    # Send the response
    await cl.Message(content=res).send()
