import chainlit as cl
from dotenv import load_dotenv
load_dotenv()
import warnings
import google.generativeai as genai
import os
from pathlib import Path as p
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
warnings.filterwarnings("ignore")
# restart python kernal if issues with langchain import.

genai.configure(api_key=os.environ.get("google_api_key"))
"""### In Context Information Retreival
"""
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0.5)

"""### Extract text from the PDF"""

pdf_loader = PyPDFLoader(r'data/data.pdf')

pages = pdf_loader.load_and_split()

prompt_template = """Your name is खोप_Bot, a knowledgeable and helpful medical assistant specializing in Immunization information. User can ask you any information related to Immunization in Nepal. Your goal is to provide accurate, clear, and supportive answers to vaccine-related questions. Replace other country name with Nepal.You'll be presented with a context and a question. Carefully examine the context to extract any relevant information that can help you answer the question precisely.Answer maynot always be in context, In that case answer relevent to the context\n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

"""### RAG Pipeline: Embedding + LLM"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

# texts

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_index = Chroma.from_texts(texts, embeddings).as_retriever()


@cl.on_message
async def main(message: cl.Message):
    docs = vector_index.get_relevant_documents(message.content)

    stuff_answer = stuff_chain(
        {"input_documents": docs, "question": message.content}, return_only_outputs=True
    )

    # Access the text content from the dictionary
    text = stuff_answer['output_text']
    await cl.Message(
        content=text,
    ).send()


