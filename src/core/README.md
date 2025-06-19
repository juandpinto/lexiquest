# Core Module Documentation

This document provides an overview of the core logic for the LexiQuest multi-agent workflow, focusing on the workflow graph defined in `graph.py` and the state management via `FullState` in `states.py`.

---

## 1. Multi-Agent Workflow Graph (`graph.py`)

The workflow is orchestrated using a directed graph (via [LangGraph](https://langchain-ai.github.io/langgraph/)), where each node represents an agent or a routing function. The graph manages the flow of information and decision-making between agents.

### Key Nodes (Agents & Router)
- **alignment_agent**: Validates user input for safety and appropriateness (e.g., using Guardrails AI). If input is invalid, the workflow ends.
- **manager**: Oversees the workflow, makes high-level decisions, and sets routing information for the next agent.
- **manager_router**: A routing function that decides which agent should act next based on the current state (typically, either `narrative_agent` or `challenge_agent`).
- **narrative_agent**: Generates the next segment of the personalized story.
- **challenge_agent**: Presents educational challenges embedded in the narrative.

### Workflow Edges
- The workflow starts at `alignment_agent`.
- If input is valid, it proceeds to `manager`, then to `manager_router`.
- `manager_router` inspects the state and routes to either `narrative_agent` or `challenge_agent`.
- After either agent acts, the workflow ends for that cycle.

### Memory
- The workflow uses a `MemorySaver` to persist state across steps.

---

## 2. State Management (`states.py`)

The global state is managed using Pydantic models, ensuring type safety and clarity.

### `FullState`
Represents the complete state of the workflow, namespaced per agent.

- **narrative** (`NarrativeState`):
  - `story`: List of story messages (excluding assessments and responses).
- **challenge** (`ChallengeState`):
  - `messages`: History of messages related to challenges.
  - `current_narrative_segment`: The current story segment.
  - `narrative_beat_info`: Key information for the current narrative beat.
  - `challenge_type`: The type of educational challenge (e.g., TILLS subtest).
  - `modality`: The modality of the challenge (e.g., text, image).
  - `story_history`: The story so far as a string.
  - `challenge_history`: List of generated challenges.
- **full_history**: Complete conversation history (all messages).
- **last_agent**: The last agent to produce output.
- **manager_decision**: The manager's routing decision.
- **input_status**: Output from the alignment agent (e.g., 'valid_input' or 'invalid_input').
- **next_agent**: The next agent to act, as determined by the router.

### State Flow
- The state is updated at each step by the corresponding agent or router node.
- Routing decisions and agent outputs are stored in the state, enabling flexible and adaptive workflows.

---

## References
- `graph.py`: Defines the workflow graph and agent orchestration logic.
- `states.py`: Defines the state models used throughout the workflow.

For further details, see the code and inline comments in each file.
