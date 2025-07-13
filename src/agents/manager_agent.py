from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel
from typing import Any, Dict
from .utils import BaseAgent
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
        challenge_index = state.narrative.challenge_index
        assessment_feedback = getattr(state, 'assessment_feedback', None)

        # If not handling challenge flow, return no decision
        if not challenge_history: # TODO: or no ACTIVE challenge
            return None, state

        # Incorporate first challenge if not already done
        if challenge_index is None:
            state.narrative.challenge_index = 0
            return {"next_agent": "narrative_agent", "task": "Incorporate the first challenge into the story."}, state

        # Check if assessment feedback is available
        if assessment_feedback:
            print(f"\n\n[Manager] Assessment feedback received: {assessment_feedback}\n")
            # TODO: Handle assessment feedback logic here

            state.assessment_feedback = None # Reset after processing

            # Continue with next challenge if available
            if len(challenge_history) > challenge_index + 1:
                state.narrative.challenge_index += 1
                # Call on the narrative agent next
                return {"next_agent": "narrative_agent", "task": "Incorporate the next challenge into the story."}, state
            else:
                return None, state


        # Save the student's response and send to assessment agent
        print(f"\n\n[Manager] Saving student response for challenge index {challenge_index}...\n")
        prev_message = state.full_history[-1]
        prev_challenge = state.challenge.challenge_history[challenge_index]
        if isinstance(prev_message, HumanMessage):
            # state.student_response = {
            #     'alphabetic': prev_message.content.strip()
            # }
            state.student_response = prev_message.content.strip()
        else:
            print("No valid student response found in the last message.")
            state.student_response = None
        state.narrative.user_responses.append({
            "triplet": prev_challenge,
            "response": state.student_response
        })

        print(f"\n\n[Manager] User responses: {state.narrative.user_responses}\n")

        # Call on the assessment agent next
        return {"next_agent": "assessment_agent", "task": "Assess the user's responses to the challenges."}, state



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
