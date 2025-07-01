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

        # Refactored: handle challenge flow in a separate method
        decision = self.handle_challenge_flow(state)
        if decision is None:
            # Default: use model to decide
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

    def handle_challenge_flow(self, state: FullState):
        """
        Handles the logic for distributing challenge triplets to the narrative agent and collecting responses.
        Returns a decision dict if handling a challenge, else None.
        """
        prev_agent = getattr(state, 'last_agent', None)
        challenge_state = getattr(state, 'challenge', None)
        challenge_history = getattr(challenge_state, 'challenge_history', []) if challenge_state else []

        # Used triplets and user responses tracking (now in narrative)
        if not hasattr(state.narrative, 'used_triplets') or state.narrative.used_triplets is None:
            state.narrative.used_triplets = []
        if not hasattr(state.narrative, 'user_responses') or state.narrative.user_responses is None:
            state.narrative.user_responses = []

        # If previous agent was challenge agent and there are triplets left
        if prev_agent == "Challenge Agent" and challenge_history:
            # Pop the first triplet
            next_triplet = challenge_history.pop(0)
            state.narrative.next_triplet = next_triplet
            state.narrative.used_triplets.append(next_triplet)
            # Remove from challenge_history
            state.challenge.challenge_history = challenge_history
            # Assign narrative agent the task
            return {"next_agent": "narrative_agent", "task": "Incorporate the following triplet into the story as a challenge: {}".format(next_triplet)}
        # If user just responded to a triplet, continue with next triplet if any
        elif prev_agent == "narrative_agent" and challenge_history:
            # Save user response if available
            if hasattr(state, 'student_response') and hasattr(state.narrative, 'next_triplet'):
                state.narrative.user_responses.append({
                    "triplet": state.narrative.next_triplet,
                    "response": state.student_response
                })
            # Pop next triplet
            next_triplet = challenge_history.pop(0)
            state.narrative.next_triplet = next_triplet
            state.narrative.used_triplets.append(next_triplet)
            state.challenge.challenge_history = challenge_history
            return {"next_agent": "narrative_agent", "task": "Incorporate the following triplet into the story as a challenge: {}".format(next_triplet)}
        # If no more triplets, send to assessment agent
        elif prev_agent == "narrative_agent" and not challenge_history:
            # Save last user response if available
            if hasattr(state, 'student_response') and hasattr(state.narrative, 'next_triplet'):
                state.narrative.user_responses.append({
                    "triplet": state.narrative.next_triplet,
                    "response": state.student_response
                })
            # Prepare expected and student responses for assessment
            state.expected_response = [item["triplet"] for item in state.narrative.user_responses]
            state.student_response = [item["response"] for item in state.narrative.user_responses]
            return {"next_agent": "assessment_agent", "task": "Assess the user's responses to the challenges."}
        # If not handling challenge flow, return None
        return None

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
