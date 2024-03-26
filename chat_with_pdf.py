import os
import json
import re
import tempfile
import streamlit as st
import langchain
from PIL import Image
from langchain.document_loaders import PyPDFLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain import LLMChain, PromptTemplate
from langchain.retrievers import EnsembleRetriever
from ragatouille import RAGPretrainedModel
from langchain.llms import Together
from ragatouille import RAGPretrainedModel
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("2005.11401.pdf")
r_docs = loader.load_and_split()

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
ragatouille_docs = [str(doc) for doc in r_docs]

RAG.index(
  collection=ragatouille_docs,
  index_name="langchain-index",
  max_document_length=512,
  split_documents=True,
)

TOGETHER_API_KEY = "5b986c28fd0eb06cac1ff36c6f900a25b33c741854fc06fbf47c6d5cbecb3aa5"

favicon = Image.open("7969d1fe6c9a25b4662a381b154fe0f4.jpg")

st.set_page_config(page_title="RAG with Mixtral 8x7B and ColBERT by Taha Efe", page_icon=favicon)
st.sidebar.image("7969d1fe6c9a25b4662a381b154fe0f4.jpg", use_column_width=True)
with st.sidebar:
  st.write("**RAG with Mixtral 8x7B and ColBERT by Taha Efe**")

os.environ["LANGCHAIN PROJECT"] = "RAG with Mixtral 8x7B and ColBERT by Taha Efe"
os.environ["LANGCHAIN_API_KEY"] = "ls__fa129a992e2b498dad13d62fa403471a"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
  # Reading documents
  docs = []
  temp_dir = tempfile.TemporaryDirectory()
  for file in uploaded_files:
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
      f.write(file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    docs.extend(loader.load())

  # Splitting the documents
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  splits = text_splitter.split_documents(docs)

  # Creating and storing embeddings
  embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
  vectordb =  Chroma.from_documents(splits, embeddings)

  # Defining the retriever
  chroma_retriever = vectordb.as_retriever(
    search_type="mmr", search_kwargs={"k":4, "fetch_k":10}
  )

  RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/langchain-index")
  ragatouille_retriever = RAG.as_langchain_retriever(k=10)
  retriever = EnsembleRetriever(retrievers=[chroma_retriever, ragatouille_retriever], weights=[0.50, 0.50])

  return retriever


uploaded_files = st.sidebar.file_uploader(
  label="Upload PDF files", type=["pdf"], accept_multiple_files = True
)
if not uploaded_files:
  st.info("Please upload PDF documents to continue.")
  st.stop()

retriever = configure_retriever(uploaded_files)

# Together API
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
llm = Together(
  model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.5,
    max_tokens=2048,
    top_k=10
)

msgs = StreamlitChatMessageHistory()

# Prompt Template
RESPONSE_TEMPLATE = """<s>[INST]
<<SYS>>
You are a helpful AI assistant.

Use the following pieces of context to answer the user's question.<</SYS>>

Anything between the following `context` html blocks is retrieved from a knowledge base.

<context>
    {context}
</context>

REMEMBER:
- If you don't know the answer, just say that you don't know, don't try to make up an answer.
- Let's take a deep breath and think step-by-step.

Question: {question}[/INST]
Helpful Answer:
"""

PROMPT = PromptTemplate.from_template(RESPONSE_TEMPLATE)
PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
  llm,
  chain_type="stuff",
  retriever=retriever,
  chain_type_kwargs={
    "verbose": True,
    "prompt": PROMPT
  }
)


if len(msgs.messages) == 0 or st.sidebar.button("New Chat"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):

        response = qa_chain({"query": user_query})

        ## Printing Answer
        answer = response["result"]
        st.write(answer)

about = st.sidebar.expander("About")
about.write("It is build by [Taha Efe Gümüş](https://www.linkedin.com/in/tahaefegumus/).")
