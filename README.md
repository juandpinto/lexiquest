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

This project implements a multi-agent system designed to engage children through interactive storytelling. The system consists of a manager agent that coordinates tasks and a narrative agent that generates personalized stories based on user input and interests.

## Project Structure

```
python-multi-agent-app
├── src
│   ├── agents
│   │   ├── __init__.py
│   │   ├── manager_agent.py
│   │   └── narrative_agent.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── graph.py
│   ├── tools
│   │   ├── __init__.py
│   │   └── handoff_tools.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd python-multi-agent-app
pip install -r requirements.txt
```

## Usage

Run the application using the following command:

```bash
python src/app.py
```

Once the application is running, you can interact with the storytelling system through the Gradio interface.

## Agents

### Manager Agent

The `ManagerAgent` class is responsible for overseeing the narrative agent. It assigns tasks, ensures that the stories generated are appropriate for children, and manages the flow of the conversation.

### Narrative Agent

The `NarrativeAgent` class generates engaging and age-appropriate stories based on user input. It takes into account the child's interests and preferences to create a personalized storytelling experience.

## Configuration

Configuration settings, including API keys and environment variables, are managed in the `src/core/config.py` file. Ensure that you set the necessary keys before running the application.
