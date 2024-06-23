import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from streamlit_chat import message
import os 

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Load the PDF file
pdf_path = "ENSAMFINAL1.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split the PDF text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

# Create embeddings for the PDF text chunks
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Create a vector store from the PDF text chunks
vectorstore = FAISS.from_documents(texts, embeddings)

# Set the page configuration
st.set_page_config(page_title="ENSAMi", page_icon="ensamii.png")

# Streamlit app
st.title("ChatBot-ENSAMi 📚👨‍🎓")

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []

def chat_with_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def conversation_chat(query):
    try:
        # Retrieve relevant documents from the vector store
        relevant_docs = vectorstore.similarity_search(query)
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Prepare the chat history for the model
        messages = [{"role": "system", "content": "Vous êtes ENSAMi, un assistant utile pour l'école ENSAM Rabat. Fournissez des informations précises et utiles sur l'école."}]
        for q, a in st.session_state["history"]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        # Add the latest query
        messages.append({"role": "user", "content": query})
        messages.append({"role": "system", "content": context})

        # Get the response from the chat model
        response = chat_with_gpt(messages)

        # Update chat history
        st.session_state["history"].append((query, response))
        return response
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        return None

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Bonjour Je suis ENSAMi! Avez-vous des questions sur l'ENSAM Rabat ?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Salut !"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Entrez votre question :", placeholder="Posez une question sur ENSAM Rabat", key='input')
            submit_button = st.form_submit_button(label='Envoyer')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                # Display user message
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                
                # Display bot response with custom avatar without emoji
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.image("ensamii.png", width=50)
                with col2:
                    st.write(st.session_state["generated"][i])

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
