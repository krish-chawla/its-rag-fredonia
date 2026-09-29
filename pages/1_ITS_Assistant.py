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
from escalation import should_escalate, show_escalation

load_dotenv()

# Same local/deployed key handling as the Flowers bot page
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# PLACEHOLDER contact info — I don't have SUNY Fredonia's actual ITS Help
# Desk phone/email, so do NOT ship this as-is. Confirm the real details
# (likely on Fredonia's IT Help Desk webpage, or ask Dr. Zubairi) and
# replace both lines below before this goes in front of real users.
CONTACT_NAME = "the ITS Help Desk"
CONTACT_LINE = "📞 [ITS Help Desk phone — 716-673-3407]  |  ✉️ [ITS Help Desk email — ITS.ServiceCenter@fredonia.edu]"

st.set_page_config(page_title="ITS Assistant", page_icon="💻")
st.title("💻 SUNY Fredonia ITS Assistant")
st.caption("Ask a question about the Acceptable Use Policy for IT resources.")


@st.cache_resource(show_spinner="Loading ITS policy...")
def build_chain():
    """Builds the RAG chain (with query rewriting) once per app session."""
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loader = TextLoader(os.path.join(dir_path, "its_policy.txt"))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    # No persist_directory here on purpose: Streamlit Cloud's filesystem is
    # ephemeral and wiped on every redeploy/restart anyway, so persisting to
    # disk buys nothing in production and just adds complexity. The vector
    # store is rebuilt fresh each time this cached function runs.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="its_policy",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Query rewriter — bridges vocabulary gaps before retrieval
    rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query rewriting assistant. Rewrite the following question
using formal vocabulary that would appear in a university IT policy document.
Use terms like: commercial use, personal financial gain, incidental personal use,
acceptable use, computing resources, authorized use.
Return only the rewritten question, nothing else.

Original question: {question}
""")
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for SUNY Fredonia's ITS department.
Answer the question based only on the following context.
If the answer is not in the context, say "I don't have that information."

Context: {context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    answer_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["rewritten"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    def full_chain(question: str) -> str:
        rewritten = rewrite_chain.invoke({"question": question})
        return answer_chain.invoke({"question": question, "rewritten": rewritten})

    return full_chain


chain = build_chain()

if "its_messages" not in st.session_state:
    st.session_state.its_messages = []

for msg in st.session_state.its_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.its_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain(user_question)
            st.markdown(response)
            if should_escalate(user_question, response):
                show_escalation(CONTACT_NAME, CONTACT_LINE)

    st.session_state.its_messages.append({"role": "assistant", "content": response})