from langchain_core.messages import SystemMessage, AIMessage
from typing import Any, Dict
from agents.utils import BaseAgent
from core.states import FullState
from pprint import pprint

# Manager agent system prompt
MANAGER_PROMPT = """
You are a manager managing the following agents:

- "narrative_agent": Assign narrative-related tasks to this agent.
- "challenge_agent": Assign challenge-related tasks to this agent.

Assign work to one agent at a time, do not call agents in parallel.

When the narrative agent provides you with a story, make sure it does not include any inappropriate elements for a child. Then return the story to the user.

The user will initiate the conversation with "--- START NOW ---", after which you should refer to the narrative agent for the first part of the story.

Your response MUST be a JSON object with two keys:
- "next_agent": either "narrative_agent" or "challenge_agent"
- "task": a string describing the task to assign

Example:
{"next_agent": "narrative_agent", "task": "Continue the story"}
"""


class ManagerAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(name='Manager Agent')
        self.model = model
        self.prompt = MANAGER_PROMPT

    def __call__(self, state: FullState) -> FullState:
        """
        Manage agents, assigning tasks.

        Accepts and returns the global FullState, updating only the relevant namespaces.
        """
        print("\n--- Running Manager Agent ---")

        # Generate the next task and agent
        decision = self.generate_task(state)
        print("Manager decision:")
        pprint(decision)
        print()

        # Add agent metadata before appending
        if isinstance(decision, AIMessage):
            decision.metadata = dict(decision.metadata or {})
            decision.metadata["agent"] = self.name

        # Store the decision in the state for the router node to use
        state.last_agent = self.name
        # Wrap the manager decision as an AIMessage for full_history
        state.full_history.append(AIMessage(content=f"Manager decision: {decision}", metadata={"agent": self.name}))

        # Attach the decision to the state for the router node
        state.manager_decision = decision

        return state

    def generate_task(self, state: FullState) -> Dict[str, str]:
        """
        Use the model to decide the next agent and task.
        Returns a dict: {"next_agent": ..., "task": ...}
        """
        # Prepare a summary of the current state for the prompt
        narrative_summary = "\n".join([msg.content for msg in state.narrative.story[-3:]])
        user_message = state.narrative.story[-1].content if state.narrative.story else "Let's start!"

        # prompt = (
        #     self.prompt
        #     + "\n\n"
        #     + f"Recent story:\n{narrative_summary}\n\n"
        #     + f"Most recent user message: {user_message}\n\n"
        # )

        # Call the model (assume it returns a stringified JSON object)
        response = self.model.invoke([SystemMessage(content=self.prompt)] + state.full_history)

        # Check if the response is a valid JSON object
        import json
        try:
            content = response.content.strip().removeprefix('```json').removesuffix('```').strip()
            decision = json.loads(content)
        except Exception:
            # Fallback: default to narrative_agent
            decision = {"next_agent": "narrative_agent", "task": "Continue the story"}

        return decision
