from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import os
import warnings
import time  # For simulating progress

print(warnings.__file__)
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Streamlit app setup
st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

# Progress bar placeholder
progress_bar = st.empty()
status_text = st.empty()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    persistent_directory = "Pdfs/chroma_db_GenAI"

    # Progress step 1: Save uploaded file temporarily
    progress_bar.progress(0)
    status_text.text("Uploading document...")
    file_path = f"Pdfs/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())  
    st.success("Document successfully uploaded!")
    progress_bar.progress(20)

    # Initialize embeddings
    status_text.text("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    progress_bar.progress(40)
    st.write("Embeddings successfully initialized.")

    # Check if the vector store exists
    if not os.path.exists(persistent_directory):
        status_text.text("Creating vector database...")
        st.info("Persistent directory does not exist. Initializing vector store...")
        progress_bar.progress(60)

        # Document Loading
        loader = PyPDFLoader(file_path=file_path)
        document = loader.load()
        st.write("Document successfully loaded.")
        progress_bar.progress(70)

        # Document Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_overlap=100, chunk_size=100)
        chunks = splitter.split_documents(documents=document)
        st.write("Document successfully chunked into smaller parts.")
        progress_bar.progress(80)

        # Creating Chroma Storage
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persistent_directory
        )
        st.success("Vector database successfully created!")
        progress_bar.progress(100)
    else:
        st.success("Vector store already exists. No need to initialize.")
        vector_store = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Create retriever
    status_text.text("Creating retriever...")
    retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )
    st.write("Retriever successfully created.")
    progress_bar.empty()
    status_text.empty()

    # Chat functionality
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("You: ", key="input")
    submit = st.button("Submit")

    if submit and query.strip():
        # Extracting similar results
        results = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in results])
        st.write(f"Extracted similar results:\n {results}\nContext:\n{context}")

        # Creating prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are an AI assistant tasked with answering questions using only the provided context. Follow these rules:"
                    + "Verify if the answer exists within the context."
                    + "If it does, respond accurately and concisely."
                    + "If it does not, state: I cannot answer this question as the context does not contain relevant information."
                ),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        st.write("Prompt template successfully created.")

        openai_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, api_key=OPENAI_API_KEY)
        chain = prompt_template | openai_model | StrOutputParser()
        response = chain.invoke({"context": results, "question": query})

        st.session_state.chat_history.append(SystemMessage(content=response))
        st.session_state.chat_history.append(HumanMessage(content=query))

        st.write(f"AI: {response}")

    # Display chat history
    st.subheader("Chat History")
    for msg in reversed(st.session_state.chat_history):
        if isinstance(msg, HumanMessage):
            st.write(f"You: {msg.content}")
        elif isinstance(msg, SystemMessage):
            st.write(f"AI: {msg.content}")
else:
    st.write("Please upload a PDF document to begin.")
