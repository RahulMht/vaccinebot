

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
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0.3)


"""### Extract text from the PDF"""

pdf_loader = PyPDFLoader(r"C:\Users\DELL\Downloads\FAQ_on_Immunization_for_Health_Workers-English.pdf")

pages = pdf_loader.load_and_split()

prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

# texts

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

#input_ question here
question = "What are vaccine hesitancy and vaccine confidence?"
docs = vector_index.get_relevant_documents(question)

stuff_answer = stuff_chain(
    {"input_documents": docs, "question": question}, return_only_outputs=True
)

# Access the text content from the dictionary
text = stuff_answer['output_text']

# Format the text into a paragraph by joining the lines with spaces
processed_output = " ".join(text.splitlines())

# Print the formatted paragraph
print(processed_output)