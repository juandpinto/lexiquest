---
title: LexiQuest
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: src/app.py
---

# LexiQuest: Personalized, Multimodal SLD Screening via Agentic Narrative AI

## Overview

LexiQuest is a multi-agent, interactive storytelling system designed to screen for Specific Learning Disorders (SLD) in children through engaging, personalized narratives and challenges. The system leverages advanced language models and a modular agent architecture to deliver a safe, adaptive, and fun experience.

## Project Structure

```
prototype/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── manager_agent.py
│   │   ├── narrative_agent.py
│   │   ├── challenge_agent.py
│   │   ├── alignment_agent.py
│   │   └── utils.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── graph.py
│   │   ├── states.py
│   │   └── challenges.py
│   └── app.py
├── data/
│   └── costs-2025.json
├── requirements.txt
├── README.md
├── flow.md
├── test.py
├── test_langsmith.py
└── test_validator.py
```

## Key Components

### Agents
- **ManagerAgent**: Oversees the workflow, delegates tasks to other agents, and ensures child-appropriate content. Decides which agent should act next based on the conversation state.
- **NarrativeAgent**: Generates engaging, age-appropriate, and personalized story segments, interacting with the user to co-create the narrative.
- **ChallengeAgent**: Presents educational challenges (e.g., vocabulary triplets) embedded in the story, adapting to the child's age and interests.
- **AlignmentAgent**: Validates user input for appropriateness using Guardrails AI, ensuring a safe and respectful environment.

### Core
- **config.py**: Handles configuration, API keys, and stores sample survey results for personalization.
- **graph.py**: Defines the multi-agent workflow using LangGraph, including routing and state management.
- **states.py**: Contains Pydantic models for global and agent-specific state.
- **challenges.py**: Defines challenge types and logic for educational tasks.

### Data
- **data/costs-2025.json**: Example data file (not directly used in core logic).

### Tests
- **test.py, test_langsmith.py, test_validator.py**: Scripts for testing various components and integrations.

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd prototype
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python src/app.py
```

Interact with the system via the Gradio web interface. You can use OpenAI or Google API keys, or fallback to Ollama if no key is provided.

## Configuration

- API keys and environment variables are managed in `src/core/config.py` and via `.env` files.
- Personalization is based on sample survey results, which can be customized in `config.py`.

## Dependencies

- [Gradio](https://gradio.app/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Guardrails AI](https://www.guardrailsai.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

See [`requirements.txt`](requirements.txt) for the full list.

To use speech to text, you'll also need [FFmpeg](https://ffmpeg.org) installed.


## License

This project is for research and prototyping purposes only.
