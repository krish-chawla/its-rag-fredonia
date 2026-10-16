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

# Real contact info for SUNY Fredonia's ITS Service Center, verified against
# fredonia.edu/about/offices/information-technology-services/contacts
# (Sept/Oct 2026). Re-check each semester in case hours change.
CONTACT_NAME = "the ITS Service Center"
CONTACT_LINE = "📞 716-673-3407  |  ✉️ ITS.ServiceCenter@fredonia.edu  |  📍 W203 Thompson Hall"
CONTACT_NOTE = (
    "Semester hours: Mon–Thu 8am–6pm, Fri 8am–4:30pm, Sat 12–5pm, closed Sun. "
    "You can also submit a ticket through Tracker (tracker.fredonia.edu)."
)

# The two knowledge-base files this bot answers from:
#   its_policy.txt       -> the formal Acceptable Use Policy
#   its_help_topics.txt  -> practical how-to topics (WiFi, laptops, email
#                           while traveling, VPN, MFA, accounts, phishing)
KNOWLEDGE_FILES = ["its_policy.txt", "its_help_topics.txt"]

st.set_page_config(page_title="ITS Assistant", page_icon="💻")
st.title("💻 SUNY Fredonia ITS Assistant")
st.caption(
    "Ask about WiFi, laptops, email and MFA while traveling, VPN setup, "
    "accounts, or the IT Acceptable Use Policy."
)

with st.expander("Example questions"):
    st.markdown(
        "- How do I connect to WiFi on campus?\n"
        "- What laptop should I buy as a computer science major?\n"
        "- Can I check my email while I'm studying abroad?\n"
        "- How do I set up the VPN?\n"
        "- How do I connect my PlayStation in my dorm?\n"
        "- My account is locked. How can I unlock it?\n"
        "- Office keeps telling me to renew my license. How do I fix it?\n"
        "- Can I use campus computers to run my own business?"
    )


@st.cache_resource(show_spinner="Loading ITS knowledge base...")
def build_chain():
    """Builds the RAG chain (with query rewriting) once per app session."""
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    documents = []
    for filename in KNOWLEDGE_FILES:
        documents.extend(TextLoader(os.path.join(dir_path, filename)).load())

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
        collection_name="its_knowledge",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Query rewriter — bridges vocabulary gaps before retrieval. It now has
    # to handle BOTH formal policy questions and casual how-to questions, so
    # it keeps the user's concrete keywords (wifi, VPN, laptop, ...) and adds
    # the matching official terms instead of replacing them.
    rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query rewriting assistant for a university IT help knowledge base.
The knowledge base has two kinds of content:
1. A formal IT Acceptable Use Policy (terms like: commercial use, personal financial
   gain, incidental personal use, acceptable use, computing resources, authorized use).
2. Practical how-to help topics (terms like: eduroam, WiFi, eServices, fredPASS,
   FREDmedia, MAC address, laptop specifications, FRED Laptop Program, Outlook,
   Microsoft 365, MFA, Microsoft Authenticator, traveling, FortiClient VPN,
   Terminal Server, phishing, ITS Service Center, ResNet, account locked, password reset,
   Microsoft Office license, unlicensed product, portal.office.com).
Rewrite the question so it keeps the user's original keywords AND adds the official
terms above that match it. Return only the rewritten question, nothing else.

Original question: {question}
""")
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for SUNY Fredonia's ITS department.
Answer the question based only on the following context.
When the answer is a procedure, give it as numbered steps.
Include exact names, addresses, and links from the context (for example vpn.fredonia.edu).
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
                show_escalation(CONTACT_NAME, CONTACT_LINE, CONTACT_NOTE)

    st.session_state.its_messages.append({"role": "assistant", "content": response})
