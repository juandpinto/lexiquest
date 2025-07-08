from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel
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

The user will initiate the conversation with "--- START NOW ---", after which you should refer to the narrative agent for the first part of the story. Make sure the challenge_agent is called every few turns.

Your response MUST be a JSON object with two keys:
- "next_agent": either "narrative_agent" or "challenge_agent"
- "task": a string describing the task to assign

DO NOT include any other text or formatting, ONLY return the JSON object.

Example:
{"next_agent": "narrative_agent", "task": "Continue the story"}
"""

# Define the structured output schema for manager decisions
class ManagerDecision(BaseModel):
    next_agent: str
    task: str


class ManagerAgent(BaseAgent):
    def __init__(self, model):
        # Wrap the model with structured output enforcement
        structured_model = model.with_structured_output(ManagerDecision)
        super().__init__(name='Manager Agent')
        self.model = structured_model
        self.prompt = MANAGER_PROMPT

    def __call__(self, state: FullState) -> FullState:
        """
        Manage agents, assigning tasks.

        Accepts and returns the global FullState, updating only the relevant namespaces.
        """
        print("\n--- Running Manager Agent ---")

        decision, state = self.handle_challenge_flow(state)
        print(f"\n\nChallenge flow decision: {decision}\n")
        if decision is None:
            # Default: use model to decide
            decision = self.generate_task(state)
            print("Manager decision:")
            pprint(decision)
            print()

        # Add agent metadata before appending
        if isinstance(decision, AIMessage):
            if getattr(decision, 'metadata', None) is None:
                decision.metadata = {}
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
        Returns a decision dict if handling a challenge, else None, along with the updated state.
        """
        challenge_history = getattr(state.challenge, 'challenge_history', [])

        # Save the student's response if active challenge
        if state.narrative.active_challenge:
            prev_message = state.full_history[-1]
            # Get previous triplet from state.narrative.story
            for i in range(len(state.narrative.story)):
                if isinstance(state.narrative.story[-i], AIMessage):
                    prev_triplet = state.narrative.story[-i].metadata.get("challenge_triplet", None)
                    break
            if isinstance(prev_message, HumanMessage):
                state.student_response = prev_message.content.strip()
            else:
                print("No valid student response found in the last message.")
                state.student_response = None
            state.narrative.user_responses.append({
                "triplet": prev_triplet,
                "response": state.student_response
            })

            print(f"\n\n[Manager] User responses: {state.narrative.user_responses}\n")

            # If active challenge but no more triplets, prepare for assessment
            if not challenge_history:
                state.narrative.active_challenge = False
                # Prepare expected and student responses for assessment
                state.expected_response = [item["triplet"] for item in state.narrative.user_responses]
                state.student_response = [item["response"] for item in state.narrative.user_responses]
                return {"next_agent": "assessment_agent", "task": "Assess the user's responses to the challenges."}, state

        # Continue with next challenge if available
        if challenge_history:
            state.narrative.active_challenge = True
            # Pop the first triplet
            next_triplet = challenge_history.pop(0)
            state.narrative.next_triplet = next_triplet
            state.narrative.used_triplets.append(next_triplet)
            # Remove from challenge_history
            state.challenge.challenge_history = challenge_history
            # Assign narrative agent the task
            return {"next_agent": "narrative_agent", "task": "Incorporate the following triplet into the story as a challenge: {}".format(next_triplet)}, state

        # If not handling challenge flow, return None
        return None, state

    def generate_task(self, state: FullState) -> dict:
        """
        Use the model to decide the next agent and task.
        Returns a dict: {"next_agent": ..., "task": ...}
        """
        narrative_summary = "\n".join([msg.content for msg in state.narrative.story[-3:]])
        user_message = state.narrative.story[-1].content if state.narrative.story else "Let's start!"

        response = self.model.invoke([SystemMessage(content=self.prompt)] + state.full_history)
        print(f"\n[Manager] Raw manager response:\n{response}\n")

        # response is now a ManagerDecision object, convert to dict
        try:
            decision = response.dict()
        except Exception:
            print("Invalid response from model, defaulting to narrative_agent.")
            decision = {"next_agent": "narrative_agent", "task": "Continue the story"}
        return decision
