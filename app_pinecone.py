from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import os
import warnings


print(warnings.__file__)
load_dotenv()


openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

file_path = "Pdfs/Generative_AI.pdf"

#3. Document Embeddings
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
print("Creted Embedding function")


# Ensure the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )


#1. Document Loading
loader = PyPDFLoader(
    file_path = file_path,
)
document = loader.load()
print("1. Successfully loaded")



#2. Document Chunking
splitter = RecursiveCharacterTextSplitter(chunk_overlap=100, chunk_size=100)
chunks = splitter.split_documents(documents = document)
print(len(chunks))
print("2. Successfully chunked")


pc = Pinecone(api_key= pinecone_api_key)
index_name = "pinecone-bot"
spec = ServerlessSpec(cloud= 'aws', region= 'us-east-1')
existing_indexes=[index_info["name"] for index_info in pc.list_indexes()]
print(existing_indexes, "\nList of Names: ", pc.list_indexes().names(), "\n current index: ", index_name)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=spec
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    # See that it is empty
    print("Index before upsert:")
    print(pc.Index(index_name).describe_index_stats())
    print("\n")

    vector_store = PineconeVectorStore(index = index_name, embedding=embeddings)
    print("4. Successfully created vector store")

    #adding Documents to vector store
    vector_store.add_documents(documents = chunks)  
    print("Successfully documents uploaded")

else:
    print("Vector store already exists. No need to initialize.")
    index = pc.Index(index_name)
    print(pc.Index(index_name).describe_index_stats())
    vector_store = PineconeVectorStore(index = index, embedding=embeddings)



#5. Retriever
retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
)
print("5. Successfully created retriever")

st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

query = st.text_input("You: ")
submit = st.button("Submit")

print(query)
query=query.strip()
print(query)

if submit and query.strip():
    print(retriever, type(retriever))
    #6. extracting similar results

    results = retriever.invoke(input = query)
    print("***********\n Results: ", results,"************\n")
    for res in results: 
        print(f"* {res.page_content} [{res.metadata}]")
    context = "\n".join([doc.page_content for doc in results])
    print(f"6. extracted similar results: \n {results}\n context: \n{context}")

    #7. Creating prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "Answer the following question only based on the below context:\n\n{context}, don't use your own knowledge.If the context didn't contain answer for the prompt,Just say sorry for the user and leave it."
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    print("7. Created Prompt Template", prompt_template)

    openai_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, api_key=openai_api_key)
    print("model is ready")
    chain = prompt_template | openai_model | StrOutputParser()
    print("chain is created")
    response = chain.invoke({"context": context, "question": query})
    print("responded")

    st.session_state.chat_history.append(SystemMessage(content=response))
    st.session_state.chat_history.append(HumanMessage(content=query))

    st.write(f"AI: {response}")


st.subheader("Chat History")
for msg in reversed(st.session_state.chat_history):  
    if isinstance(msg, HumanMessage):
        st.write(f"You: {msg.content}")
    elif isinstance(msg, SystemMessage):
        st.write(f"AI: {msg.content}")

st.subheader("Debug Chat History")
st.write(st.session_state.chat_history)

