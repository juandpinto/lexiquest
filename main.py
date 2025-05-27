# %%
import getpass
import os

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    dotenv_path = os.path.join(os.getcwd(), ".env")
    load_dotenv(dotenv_path=dotenv_path)
    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"


# %%
survey_results = {
    'age': 7,
    'interests': ['space', 'dinosaurs'],
    'wants_to_be': 'astronaut',
    'favorite_color': 'blue',
    'favorite_food': 'pizza',
    'favorite_animal': 'dolphin',
    'favorite_book': 'Harry Potter',
    'favorite_movie': 'Toy Story',
    'favorite_subject': 'science',
}


# %%
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


model = ChatOllama(
    model="gemma3",
    temperature=0.8,
    # other params...
)


# %%
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a master storyteller who really understands how to engage children. Co-create a simple short story with the child, using short sentences and simple words. The story should be fun, exciting, age-appropriate, and personalized to their interests. The child is {age} years old, wants to be a {wants_to_be} when they grow up, and has the following interests:

            General interests: {interests}.
            Favorite color: {favorite_color}.
            Favorite food: {favorite_food}.
            Favorite animal: {favorite_animal}.
            Favorite book: {favorite_book}.
            Favorite movie: {favorite_movie}.
            Favorite subject: {favorite_subject}.

            Begin by setting the scene, organically asking the child questions to guide the story, and responding to their answers. Use emojis to make it more engaging!""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# %%
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    age: int
    interests: Sequence[str]
    wants_to_be: str
    favorite_color: str
    favorite_food: str
    favorite_animal: str
    favorite_book: str
    favorite_movie: str
    favorite_subject: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# %%
config = {"configurable": {"thread_id": "a"}}

# query = "Let's start!"

# input_data = survey_results.copy()
# input_data['messages'] = [HumanMessage(query)]

# # input_messages = [HumanMessage(query)]

# output = app.invoke(
#     input_data,
#     config
# )

# # output["messages"][-1].pretty_print()
# print(output["messages"][-1].content)


# # %%
# query = ""

# while query != "exit":
#     # Get user input
#     print()
#     print()
#     query = input(">>> ")

#     if query.lower() == "exit":
#         print("Exiting the conversation.")
#         break

#     # query = "Let's start."

#     input_messages = [HumanMessage(query)]
#     for chunk, metadata in app.stream(
#         {"messages": input_messages},
#         config,
#         stream_mode="messages",
#     ):
#         if isinstance(chunk, AIMessage):  # Filter to just model responses
#             print(chunk.content, end="")

#     print()


# %%
