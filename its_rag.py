import shutil
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Load the ITS policy document
loader = TextLoader("its_policy.txt")
documents = loader.load()

# Chunk it
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print(f"Document split into {len(chunks)} chunks")

# Embed and store in ChromaDB
embeddings = OpenAIEmbeddings()

# Clear old vector store so it rebuilds with updated document
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="its_policy",
    persist_directory="./chroma_db"
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

# Build chain using modern LangChain syntax
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for SUNY Fredonia's ITS department.
Answer the question based only on the following context.
If the answer is not in the context, say "I don't have that information."

Context: {context}

Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["rewritten"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Test questions
test_questions = [
    "Can I share my password with a friend?",
    "What happens if I violate the acceptable use policy?",
    "Am I allowed to use Fredonia computers for personal business?",
    "Can Fredonia monitor my account without telling me?",
    "What counts as excessive use of computing resources?"
]

print("\n" + "="*60)
print("ITS FREDONIA POLICY CHATBOT — TEST RUN")
print("="*60)

for question in test_questions:
    print(f"\nQ: {question}")
    rewritten = rewrite_chain.invoke({"question": question})
    answer = chain.invoke({"question": question, "rewritten": rewritten})
    print(f"A: {answer}")
    print("-"*60)