import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dir, "company_docs.txt")

print(f"Loading from: {path}")
print(f"File exists: {os.path.exists(path)}")

loader = TextLoader(path)
documents = loader.load()
print(f"\nRaw document content:\n{documents[0].page_content}")

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
print(f"\nNumber of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}: '{chunk.page_content}'")