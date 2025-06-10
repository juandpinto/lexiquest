from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from agents.narrative_agent import formatted_narrative_prompt
from agents.manager_agent import manager_system_prompt
from tools.handoff_tools import assign_to_narrative_agent

def initialize_graph(llm):
    # Create narrative agent
    narrative_agent_runnable = create_react_agent(
        model=llm,
        prompt=formatted_narrative_prompt, # Using the imported formatted prompt
        name="narrative_agent",
        tools=[], # Narrative agent might have its own tools in the future
    )

    # Create manager agent
    manager_agent_runnable = create_react_agent(
        model=llm,
        tools=[assign_to_narrative_agent], # Tool for manager to delegate
        prompt=manager_system_prompt, # Using the imported manager prompt
        name="manager",
    )

    # Initialize memory
    memory = MemorySaver()

    # Define the multi-agent supervisor graph
    supervisor = (
        StateGraph(MessagesState)
        .add_node("manager", manager_agent_runnable) # Ensure node name matches agent name and handoff tool target
        .add_node("narrative_agent", narrative_agent_runnable) # Ensure node name matches agent name
        .add_edge(START, "manager")
        # Manager can delegate to narrative_agent, narrative_agent returns to manager
        .add_edge("narrative_agent", "manager")
        # Define how the manager decides to end or loop (implicitly handled by REACT agent's tool calling or final answer)
        # If manager's response is the final one, it can go to END.
        # This might require conditional edges or further refinement in manager's logic/prompt
        # For now, let's assume the manager can decide to END.
        # We need to define destinations for the manager if it's not just looping or handing off.
        # The original code had .add_node(manager_agent, destinations=("narrative_agent", END))
        # This implies the manager itself can route to narrative_agent or END.
        # Let's adjust the manager node definition slightly if that's the intent.
        # However, create_react_agent returns a Runnable, not a node definition with explicit destinations.
        # The destinations are typically handled by the graph's .add_conditional_edges or by tools returning Command(goto=...).
        # The handoff tool already handles the goto.
        # The manager agent, if it decides the conversation is over, should produce a final response.
        # The graph implicitly goes to END when the current agent produces a final response rather than calling a tool.
        .compile(checkpointer=memory)
    )
    # To explicitly allow manager to end:
    # supervisor.add_conditional_edges("manager", lambda x: END if <condition_for_end> else "narrative_agent")
    # But for now, the REACT agent framework handles this.

    return supervisor
