from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

conversation = [
    SystemMessage(content="You are a helpful customer service assistant for a small business called Fredonia Flowers."),
]

print("Chat with Fredonia Flowers bot (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    conversation.append(HumanMessage(content=user_input))
    response = llm.invoke(conversation)
    conversation.append(AIMessage(content=response.content))
    
    print(f"Bot: {response.content}\n")