import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os 

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Load the PDF file
pdf_path = "ENSAMr.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split the PDF text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

# Create embeddings for the PDF text chunks
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Create a vector store from the PDF text chunks
vectorstore = FAISS.from_documents(texts, embeddings)

# Streamlit app
st.title("School Chatbot")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def chat_with_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=messages
    )
    return response['choices'][0]['message']['content']

query = st.text_input("Enter your query:")
if query:
    try:
        # Retrieve relevant documents from the vector store
        relevant_docs = vectorstore.similarity_search(query)
        context = " ".join([doc.page_content for doc in relevant_docs])
        
        # Prepare the chat history for the model
        messages = [{"role": "system", "content": "You are ENSAMI, a helpful assistant for ENSAM school. Provide accurate and helpful information about the school."}]
        for q, a in st.session_state["chat_history"]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        
        # Add the latest query
        messages.append({"role": "user", "content": query})
        messages.append({"role": "system", "content": context})
        
        # Get the response from the chat model
        response = chat_with_gpt(messages)
        
        # Update chat history
        st.session_state["chat_history"].append((query, response))
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the chat history
for query, response in st.session_state["chat_history"]:
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Chatbot:** {response}")
