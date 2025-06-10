from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.prompts import ChatPromptTemplate

from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver


# Sample survey results
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

# Generate dynamic narrative agent system prompt
prompt_template_str = ChatPromptTemplate.from_messages(
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
        # MessagesPlaceholder(variable_name="messages"),
    ]
)

completed_prompt = prompt_template_str.format(**survey_results)


# Set up agent communication
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool


# Handoffs
assign_to_narrative_agent = create_handoff_tool(
    agent_name="narrative_agent",
    description="Assign task to a narrative agent that is a master storyteller.",
)


def initialize_graph(llm):
    # Create narrative agent
    narrative_agent = create_react_agent(
        model=llm,
        prompt=completed_prompt,
        name="narrative_agent",
        tools=[],
    )

    # Create manager agent
    manager_agent = create_react_agent(
        model=llm,
        tools=[assign_to_narrative_agent],
        prompt=(
"""You are a manager managing the following agents:

    - A narrative agent. Assign narrative-related tasks to this agent.

Assign work to one agent at a time, do not call agents in parallel.

When the narrative agent provides you with a story, make sure it does not include any inappropriate elements for a child. Then return the story to the user.

The user will initiate the conversation with "Let's start!", after which you should refer to the narrative agent for the first part of the story."""
        ),
        name="manager",
    )


    # Initialize memory
    memory = MemorySaver()


    # Define the multi-agent supervisor graph
    supervisor = (
        StateGraph(MessagesState)
        .add_node(manager_agent, destinations=("narrative_agent", END))
        .add_node(narrative_agent)
        .add_edge(START, "manager")
        # Always return back to the manager
        .add_edge("narrative_agent", "manager")
        # Add more agents here
        .compile(checkpointer=memory)
    )

    return supervisor
