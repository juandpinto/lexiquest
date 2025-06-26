from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents import NarrativeAgent, ChallengeAgent, ManagerAgent, AlignmentAgent

from core.config import survey_results
from core.states import FullState


def initialize_graph(llm):
    # Create agents
    manager_agent = ManagerAgent(model=llm)
    narrative_agent = NarrativeAgent(model=llm, survey_results=survey_results)
    challenge_agent = ChallengeAgent(model=llm)
    alignment_agent = AlignmentAgent()

    def survey_router(state: FullState) -> FullState:
        if state.input_status == 'valid_input':
            if state.narrative.finished_survey:
                return 'manager'
            else:
                return 'narrative_agent'
        else:
            return END

    # Router node: sets a routing key in the state for conditional routing
    def manager_router(state: FullState) -> FullState:
        print("[manager_router] Input state:", state)
        decision = getattr(state, "manager_decision", None)
        if not decision or not isinstance(decision, dict):
            state.next_agent = "narrative_agent"
        else:
            state.next_agent = decision.get("next_agent", "narrative_agent")
        print("[manager_router] Output state:", state)
        print("[manager_router] Output type:", type(state))
        return state

    # Initialize memory
    memory = MemorySaver()

    # Define the multi-agent workflow graph
    workflow = (
        StateGraph(FullState)
        .add_node('alignment_agent', alignment_agent)
        .add_node('manager', manager_agent)
        .add_node('narrative_agent', narrative_agent)
        .add_node('challenge_agent', challenge_agent)
        .add_node('manager_router', manager_router)

        .add_edge(START, 'alignment_agent')
        .add_conditional_edges(
            'alignment_agent',
            lambda state: END if state.input_status =='invalid_input' else 'manager' if state.narrative.finished_survey else 'narrative_agent'
            # # Use the value of state.input_status for routing
            # lambda state: getattr(state, 'input_status', 'valid_input'),
            # {
            #     'invalid_input': END,
            #     'valid_input': 'manager',
            # }
        )
        .add_edge('manager', 'manager_router')
        .add_conditional_edges(
            'manager_router',
            lambda state: getattr(state, 'next_agent', 'narrative_agent'),
            {
                'narrative_agent': 'narrative_agent',
                'challenge_agent': 'challenge_agent',
            }
        )
        .add_edge('narrative_agent', END)
        .add_edge('challenge_agent', 'manager')

        .compile(checkpointer=memory)
    )

    return workflow
