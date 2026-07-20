from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def structure_prompt(unstructured_prompt: str) -> str:
    system = SystemMessage(content="""You are a prompt engineering assistant.
Your job is to analyze an unstructured prompt and return a fully structured version of it.

You must identify and explicitly label the following five components:
1. TASK — What is the AI being asked to do?
2. PERSONA/ROLE — What role should the AI adopt?
3. CONSTRAINTS — What rules, limitations, or requirements apply?
4. EXPECTED OUTPUT FORMAT — How should the response be structured?
5. CONFIDENCE LEVEL — How confident are you in each component you identified? (0-100%)

Return your response in this exact format:
TASK: ...
PERSONA/ROLE: ...
CONSTRAINTS: ...
EXPECTED OUTPUT FORMAT: ...
CONFIDENCE LEVEL: Task (X%), Persona (X%), Constraints (X%), Output Format (X%)""")

    human = HumanMessage(content=f"Structure this prompt:\n\n{unstructured_prompt}")

    response = llm.invoke([system, human])
    return response.content

# Three spring prompts
spring_prompts = {
    "Tutoring Planner": """You are an AI tutoring planner for an undergraduate Computer Science course.
Task: Design a structured tutoring plan for a student struggling with Heaps and their usage in programming.
Student Profile: Junior CS major, below-average performance in heap-related questions, Strengths: Arrays, recursion, Weaknesses: Pointers, tree traversal logic, 3 sessions 60 minutes each, one-on-one tutoring.""",

    "Project Manager": """You are an AI project manager.
Task: Create a 6-week project plan for developing a customer account system for a company website.
Include timeline, work breakdown structure, team roles, working hours estimation, handling of unexpected scenarios, and features: user authentication, profile management, customer points/rewards system.""",

    "Travel Planner": """You are an AI travel planner.
Task: Create a detailed 10-day Switzerland travel itinerary.
Include day-by-day schedule, attractions, logical travel routes, walking directions, photo opportunities, food recommendations, and contingency plans for bad weather and delays."""
}

for name, prompt in spring_prompts.items():
    print(f"\n{'='*60}")
    print(f"PROMPT: {name}")
    print(f"{'='*60}")
    print(structure_prompt(prompt))