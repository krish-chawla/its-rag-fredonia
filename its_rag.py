import shutil
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import escalation
from escalation import Ticket, check_after_answer, check_before_answer, handoff_message, route_ticket

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
If the answer is not in the context, reply with exactly: "{no_answer}"

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

answer_chain = prompt | llm | StrOutputParser()


def answer_question(question):
    """Answer a question, deciding along the way whether a human needs to take over.

    Returns a dict with the answer (None if the bot shouldn't answer), the
    escalation decision, the retrieved source snippets, and the top relevance score.
    """
    if escalation.wants_human(question):
        # No point spending API calls on retrieval — go straight to handoff
        return {"answer": None, "decision": check_before_answer(question, None),
                "sources": [], "top_score": None}

    rewritten = rewrite_chain.invoke({"question": question})
    results = vectorstore.similarity_search_with_relevance_scores(rewritten, k=6)
    docs = [doc for doc, _ in results]
    top_score = max((score for _, score in results), default=None)
    sources = [doc.page_content[:80].replace("\n", " ") + "..." for doc in docs[:3]]

    decision = check_before_answer(question, top_score)
    answer = None
    if not decision.escalate or decision.answer_first:
        answer = answer_chain.invoke({
            "context": format_docs(docs),
            "question": question,
            "no_answer": escalation.NO_ANSWER_SENTINEL,
        })
        if not decision.escalate:
            decision = check_after_answer(answer)
        elif escalation.is_no_answer(answer):
            answer = None  # already handing off; don't show an unhelpful reply first

    return {"answer": answer, "decision": decision, "sources": sources, "top_score": top_score}


def prompt_for_contact():
    name = input("Your name: ").strip()
    while True:
        email = input("Your email (so ITS can reply): ").strip()
        if escalation.is_valid_email(email):
            return name, email
        print("That doesn't look like a valid email address — please try again.")


def hand_off(question, result, transcript):
    decision = result["decision"]
    print(f"Bot: {handoff_message(decision)}")
    if decision.reason != escalation.USER_REQUESTED:
        if input("Send this to the ITS Service Desk? (y/n): ").strip().lower() not in ("y", "yes"):
            print("Bot: No problem. Let me know if there's anything else I can help with.\n")
            return
    name, email = prompt_for_contact()
    ticket = Ticket(
        reason=decision.reason,
        priority=decision.priority,
        question=question,
        contact_name=name,
        contact_email=email,
        bot_answer=result["answer"],
        sources=result["sources"],
        transcript=list(transcript),
    )
    delivered = route_ticket(ticket)
    if "email" in delivered:
        print(f"Bot: Done — your ticket number is {ticket.id}. ITS staff will reply to {email}.\n")
    elif delivered:
        print(f"Bot: I've logged your request as {ticket.id}. ITS staff will follow up at {email}.\n")
    else:
        print("Bot: Sorry, I couldn't submit that. Please contact the ITS Service Desk directly.\n")


def run_tests():
    test_questions = [
        "Can I share my password with a friend?",
        "What happens if I violate the acceptable use policy?",
        "Am I allowed to use Fredonia computers for personal business?",
        "Can Fredonia monitor my account without telling me?",
        "What counts as excessive use of computing resources?",
        # Escalation cases
        "Can I talk to a real person?",
        "I think my account was hacked, someone is using my email",
        "How do I connect my Xbox to the campus wifi?",
    ]

    print("\n" + "="*60)
    print("ITS FREDONIA POLICY CHATBOT — TEST RUN")
    print("="*60)

    for question in test_questions:
        print(f"\nQ: {question}")
        result = answer_question(question)
        decision = result["decision"]
        if result["answer"]:
            print(f"A: {result['answer']}")
        score = result["top_score"]
        print(f"   top relevance: {score:.3f}" if score is not None else "   top relevance: n/a")
        if decision.escalate:
            print(f"   ESCALATE -> {decision.reason} (priority: {decision.priority})")
        print("-"*60)


def chat():
    print("\nChat with the ITS policy bot (type 'quit' to exit)\n")
    transcript = []
    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        transcript.append({"role": "user", "content": question})
        result = answer_question(question)
        if result["answer"]:
            print(f"Bot: {result['answer']}\n")
            transcript.append({"role": "bot", "content": result["answer"]})
        if result["decision"].escalate:
            hand_off(question, result, transcript)


if __name__ == "__main__":
    run_tests()
    chat()
