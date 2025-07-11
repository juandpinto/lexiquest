from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage

from agents import NarrativeAgent, ChallengeAgent, ManagerAgent, AlignmentAgent, AssessmentAgent

from core.config import survey_results
from core.states import FullState


def finish_survey_node(state: FullState) -> FullState:
    """
    If the survey is finished, yield a message to the frontend before the narrative agent runs.
    """
    print("[finish_survey_node] Checking if survey is finished...")
    if getattr(state.narrative, "finished_survey", False):
        print("[finish_survey_node] Survey is finished, resetting story and full history.")
        separator_message = HumanMessage(content="--- START NOW ---")
        state.full_history = [separator_message]  # Reset full history to just the separator
        state.narrative.story = [separator_message]  # Reset full history to just the separator
        print("[finish_survey_node] Story and full history reset.")
    return state


def initialize_graph(llm):
    # Create agents
    manager_agent = ManagerAgent(model=llm)
    narrative_agent = NarrativeAgent(model=llm, survey_results=survey_results)
    challenge_agent = ChallengeAgent(model=llm)
    assessment_agent = AssessmentAgent(model=llm)
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
        print("[manager_router] Input state:", state.manager_decision)
        decision = getattr(state, "manager_decision", None)
        if not decision or not isinstance(decision, dict):
            state.next_agent = "narrative_agent"
        else:
            state.next_agent = decision.get("next_agent", "narrative_agent")
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
        .add_node('assessment_agent', assessment_agent)
        .add_node('finish_survey_node', finish_survey_node)  # Add skip indicator node

        .add_edge(START, 'alignment_agent')
        .add_conditional_edges(
            'alignment_agent',
            lambda state: 'END' if state.input_status == 'invalid_input' else 'manager' if state.narrative.finished_survey else 'narrative_agent',
            {
                'manager': 'manager',
                'narrative_agent': 'narrative_agent',
                'END': END,
            }
        )
        .add_edge('manager', 'manager_router')
        .add_conditional_edges(
            'manager_router',
            lambda state: getattr(state, 'next_agent', 'narrative_agent'),
            {
                'narrative_agent': 'narrative_agent',
                'challenge_agent': 'challenge_agent',
                'assessment_agent': 'assessment_agent',
            }
        )
        # Insert skip indicator before narrative_agent if survey was skipped
        .add_conditional_edges(
            'narrative_agent',
            lambda state: 'finish_survey_node' if state.narrative.story[-1].content == '**--- BEGINNING STORY ---**\n---\n' else 'END',
            {
                'finish_survey_node': 'finish_survey_node',
                'END': END,
            }
        )
        .add_edge('finish_survey_node', 'narrative_agent')
        .add_edge('challenge_agent', 'manager')
        .add_edge('assessment_agent', 'manager')

        .compile(checkpointer=memory)
    )

    return workflow
