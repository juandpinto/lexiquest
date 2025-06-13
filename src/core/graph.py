from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.narrative_agent import NarrativeAgent
from agents.challenge_agent import ChallengeAgent
from agents.manager_agent import ManagerAgent

from core.config import survey_results
from core.states import FullState

def initialize_graph(llm):
    # Create agents
    manager_agent = ManagerAgent(model=llm)
    narrative_agent = NarrativeAgent(model=llm, survey_results=survey_results)
    # challenge_agent = ChallengeAgent(model=llm)
    challenge_agent = lambda state: print(f"Challenge agent is not implemented yet. Here is the state it received:\n{state}")

    # Router node: decides which agent to call next based on manager_decision in state
    def manager_router(state: FullState):
        decision = getattr(state, "manager_decision", None)
        if not decision or not isinstance(decision, dict):
            # Default to narrative_agent if no decision
            return "narrative_agent"
        return decision.get("next_agent", "narrative_agent")

    # Initialize memory
    memory = MemorySaver()

    # Define the multi-agent supervisor graph
    supervisor = (
        StateGraph(FullState)
        .add_node('manager', manager_agent)
        .add_node('manager_router', manager_router)
        .add_node('narrative_agent', narrative_agent)
        .add_node('challenge_agent', challenge_agent)

        .add_edge(START, 'manager')
        .add_edge('manager', 'manager_router')
        .add_conditional_edges(
            'manager_router',
            {
                'narrative_agent': 'narrative_agent',
                'challenge_agent': 'challenge_agent',
            },
            default='narrative_agent',
        )
        .add_edge('narrative_agent', END)
        .add_edge('challenge_agent', END)
        .compile(checkpointer=memory)
    )

    return supervisor
