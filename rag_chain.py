import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Step 1: Load the document
dir = os.path.dirname(os.path.abspath(__file__))
loader = TextLoader(os.path.join(dir, "company_docs.txt"))
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: Embed and store in ChromaDB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    chunks, 
    embeddings,
    collection_name="fredonia_flowers",
    persist_directory=None
)
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 4: Build prompt template
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

# Step 5: Build the chain
llm = ChatOpenAI(model="gpt-4o-mini")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6: Run test questions first
test_questions = [
    "Do you do wedding flowers?",
    "What if my delivery arrives damaged?",
    "Can I get flowers delivered outside Fredonia?",
    "Do you offer subscriptions?",
    "How do I take care of my flowers?",
    "Can I order online?",
    "Do you take Apple Pay?"
]

print("="*60)
print("FREDONIA FLOWERS CHATBOT — TEST RUN")
print("="*60)

for question in test_questions:
    print(f"\nQ: {question}")
    response = chain.invoke(question)
    print(f"A: {response}")
    print("-"*60)

# Step 7: Interactive loop
print("\nNow entering interactive mode (type 'quit' to exit)\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    response = chain.invoke(question)
    print(f"Bot: {response}\n")