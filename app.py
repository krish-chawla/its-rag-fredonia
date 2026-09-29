import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# On Streamlit Community Cloud, the OpenAI key comes from st.secrets instead
# of a local .env file. Locally, there is no secrets.toml file at all, and
# st.secrets raises an error the moment it's touched in that case — so this
# is wrapped in a try/except to fail silently and fall back to the .env
# value that load_dotenv() already loaded above.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Fredonia Flowers Assistant", page_icon="🌷")
st.title("🌷 Fredonia Flowers — Customer Service Assistant")
st.caption("Ask a question about our flowers, orders, delivery, or policies.")


@st.cache_resource(show_spinner="Loading knowledge base...")
def build_chain():
    """Builds the RAG chain once per app session and caches it, so the
    document isn't re-embedded on every single question."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    loader = TextLoader(os.path.join(dir_path, "company_docs.txt"))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="fredonia_flowers",
        persist_directory=None,
    )
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful customer service assistant for Fredonia Flowers.
Use the context below to answer the customer's question.
If the context contains relevant information, use it to give a helpful answer.
If the question is general and you can reasonably answer it from the context or common sense, do so.
Only say "I don't have that information" if the question is completely unrelated to the business.

Context: {context}

Question: {question}

Answer:
""")

    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


chain = build_chain()

# Keep chat history across reruns (Streamlit reruns the whole script on
# every interaction, so session_state is what makes messages persist)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render the existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom of the page
user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(user_question)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})