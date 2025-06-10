# Manager agent system prompt
manager_system_prompt = """You are a manager managing the following agents:

    - A narrative agent. Assign narrative-related tasks to this agent.

Assign work to one agent at a time, do not call agents in parallel.

When the narrative agent provides you with a story, make sure it does not include any inappropriate elements for a child. Then return the story to the user.

The user will initiate the conversation with "Let's start!", after which you should refer to the narrative agent for the first part of the story."""


class ManagerAgent:
    def __init__(self, narrative_agent):
        self.narrative_agent = narrative_agent

    def assign_task(self, task):
        # Assign a narrative-related task to the narrative agent
        print(f"Assigning task to narrative agent: {task}")
        self.narrative_agent.receive_task(task)

    def review_story(self, story):
        # Ensure the story is appropriate for children
        if self.is_story_appropriate(story):
            print("Story is appropriate for children.")
            return story
        else:
            print("Story contains inappropriate elements.")
            return "The story is not suitable for children."

    def is_story_appropriate(self, story):
        # Placeholder for story appropriateness check
        inappropriate_keywords = ["violence", "death", "scary"]
        return not any(keyword in story for keyword in inappropriate_keywords)
