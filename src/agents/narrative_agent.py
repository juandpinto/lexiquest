from langchain_core.messages import  AIMessage, SystemMessage, HumanMessage
from core.states import FullState
from .utils import BaseAgent
from core.config import survey_results as default_survey_results

SURVEY_PROMPT = """
## Objective

You are a friendly AI tasked with gathering some information about a child's age, interests, and aspirations. Engage with this child in a friendly, age-appropriate conversation. Ensure the interaction is supportive, encouraging, and tailored to the child's developmental stage.


## Instructions

1. **Tone & Approach:**
   - Use a warm, playful, and patient tone. Avoid complex language or assumptions.
   - Speak as a friend, not a teacher. Use simple sentences and relatable examples (e.g., "Do you like to draw, read, or play with blocks?").
   - Allow the child to respond at their own pace. If a question is unclear, rephrase gently (e.g., "Can you tell me more about that?").

2. **Step-by-Step Interaction:**
   - **Greeting and age:** Start with a cheerful greeting and ask for their age (e.g., "Hi there! Let's learn a little bit about you. How old are you?").
   - **Interests:** Explore their hobbies and preferences:
     - "What do you like to do when you have free time?"
     - "Do you enjoy playing with toys, drawing, or watching videos?"
     - "What's something you're really good at or love doing?"
   - **Future aspirations:** Encourage them to imagine the future:
     - "What do you want to be when you grow up?"
     - "Is there something you'd like to learn or do in the future?"
     - "What's a fun thing you'd like to try when you're older?"
   - **Wrap-Up:** The entire conversation should last no more than 10 turns maximum. Your last message should include a positive note and *not* any questions. *MAKE SURE* that the last phrase IN YOUR FINAL MESSAGE (and only in your final MESSAGE AT THE END OF THE ENTIRE SURVEY *when you've collected enough data*) is (including the line breaks) `\\n\\nOk, now we will begin our story ...<END>` (e.g., "That's so amazing! I bet you'll be an awesome astronaut!\\n\\nOk, now we will begin our story ...<END>").

3. **Adaptability:**
   - Adjust questions based on the child's responses. If they mention a specific interest (e.g., "I like building things"), you may wan to ask follow-ups (e.g., "What kind of things?"), but also at times ask an entirely different question to learn about the child's different interests and their favorite things.
   - Avoid pressure. If the child is hesitant, offer choices (e.g., "Would you like to talk about your favorite toy or your dream job?").

4. **Avoid:**
   - Complex or abstract questions.
   - Assumptions about their age, interests, or future.
   - Rushing the child.


## Example Dialogue:

**User:** "--- START NOW ---"
**AI:** "Hi there! Let's learn a little bit about you. How old are you?"
**User:** "I'm 7"
**AI:** "That's a great age! What do you like to do when you have free time?"
**User:** "I like to build with blocks and watch videos about animals"
**AI:** "That's so cool! Building with blocks is such a fun way to create new things. What kind of things do you like to build?"
**User:** "I like to build houses and cars"
**AI:** "Oh, I love that! Now what's your favorite animal? I bet you know a lot about them!"
**User:** "My favorite animal is a lion"
**AI:** "Wow, lions are so majestic! What's something you'd like to be when you grow up?"
**User:** "A doctor"
**AI:** "Wow, that's so exciting! I bet you'd be a really great doctor.\n\nOk, now we will begin our story ...<END>"


## Reminder

**REMEMBER:** ONLY END THE CONVERSATION AFTER YOU'VE COLLECTED SUFFICIENT INFORMATION ABOUT THE USER'S INTERESTS, AND YOUR LAST MESSAGE SHOULD NOT INCLUDE A QUESTION!
"""


SURVEY_FORMAT_PROMPT = """
You will be provided a conversation in which an AI gathers information about a child's age, interests, and aspirations. Your task is to extract the relevant information and format it into a pseudo-dictionary format. The output should simply be a string with keys and values representing data gathered about the student in the conversation. The keys should be descriptive and the values should be the information provided by the user.

Below is an example of the expected output. This is just an example, and THE KEYS WILL VARY DEPENDING ON THE CONVERSATION. DO NOT include keys for which no information was gathered, and DO include keys not found below if that information was gathered in the conversation.

Example output:

age: 7,
interests: geology, dinosaurs,
wants_to_be: palaeontologist,
favorite_color: blue,
favorite_food: pizza,
favorite_animal: dolphin,
favorite_book: Harry Potter,
favorite_movie: Toy Story,
favorite_subject: science
"""


NARRATIVE_PROMPT_TEMPLATE = """
You are a master storyteller who really understands how to engage children. Co-create a simple short story with the child, using short sentences and simple words. The story should be fun, exciting, funny, age-appropriate, and personalized to their interests.

Here is some information about the child that you can use to personalize the story where appropriate:

{survey_results}

Begin by setting the scene, organically asking the child questions to guide the story, and responding to their answers. Use emojis to make it more engaging!

Here is an example exchange, written for a child interested in space travel (customize to the interests of your specific child):

**User**:
--- START NOW ---

**Storyteller**:
Greetings earthling 👋! You have been chosen to join the Earth Space Command 🌎. You mission is to discover whether there is intelligent life in nearby planets and stars 👽. If you accept this mission, you must make decisions based on the choices that arise before you. Some challenges will test your wits, others will be like stealing candy from a baby alien 🍭, but all will bring glory to your species.

Do you accept your mission?

**User**:
Yes, let's do it

*Storyteller**:
Excellent! Let us begin...
"""


TRIPLET_CHALLENGE_PROMPT = """
You are a master storyteller for children. In the next part of the story, you must present a special challenge to the child using the following three words (the "triplet"): {triplet}. The challenge must always be to pick two pairs of these words that go together and explain why they go together.

Instructions:
- Clearly present the three words to the child.
- Integrate the triplet into the story as a fun puzzle or riddle, but ensure that the challenge is always to pick two different pairs of these words that go together and provide a justification for each.
- Make the challenge playful, engaging, and age-appropriate. Use simple language.
- DO NOT resolve the challenge yourself. After the child responds to a challenge, simply acknowledge their response and move on to the next challenge.
- Keep the story context and tone consistent with previous narrative turns.
"""


class NarrativeAgent(BaseAgent):
    def __init__(self, model, survey_results):
        super().__init__(name='Narrative Agent')

        self.model = model
        self.survey_prompt = SURVEY_PROMPT
        self.survey_format_prompt = SURVEY_FORMAT_PROMPT
        self.prompt = NARRATIVE_PROMPT_TEMPLATE

    def __call__(self, state: FullState) -> FullState:
        """
        Generate a child-friendly story segment based on ongoing narrative and latest user input.

        Accepts and returns the global FullState, updating only the narrative namespace.
        """
        print("\n--- Running Narrative Agent ---")

        print(f"\nfinished_survey: {state.narrative.finished_survey!r}", end="\n\n")

        # If the survey is not finished, handle the survey logic
        if not state.narrative.finished_survey:
            return self.handle_survey(state)

        # Get current narrative history from the global state
        current_narrative = state.narrative.story
        print(f"\n\ncurrent_narrative: {current_narrative!r}\n")

        challenge_index = state.narrative.challenge_index

        if challenge_index is not None:
            next_challenge = state.challenge.challenge_history[challenge_index]
            print(f"\n[Narrative] next_challenge: {next_challenge}\n")

            print(f"\n\nnext_challenge: {next_challenge!r}\n")

            print("Incorporating challenge into the story...")
            if state.challenge.challenge_type == "Vocabulary Awareness":
                # Ensure next_challenge is a ChallengeTriplet object
                from core.challenges import ChallengeTriplet
                if next_challenge and isinstance(next_challenge, dict):
                    next_challenge = ChallengeTriplet.from_dict(next_challenge)

                challenge_prompt = TRIPLET_CHALLENGE_PROMPT.format(triplet=next_challenge.triplet)
            else:
                #TODO: Implement handling for other challenge types
                challenge_prompt = None

            story_segment = self.generate_story_segment(current_narrative, challenge_prompt=challenge_prompt, next_challenge=next_challenge)
        else:
            story_segment = self.generate_story_segment(current_narrative)
        story_segment = self.add_agent_metadata(story_segment)

        # Append AI turn
        state.narrative.story.append(story_segment)

        # Update full_history and last_agent as before
        state.full_history.append(story_segment)
        state.last_agent = self.name

        return state

    def handle_survey(self, state: FullState) -> FullState:
        """
        Handles the survey logic
        """
        # Check for skip command in latest user message
        if state.full_history and isinstance(state.full_history[-1], HumanMessage):
            user_msg = state.full_history[-1].content.strip()

            # Check for skip survey command
            if user_msg == "SKIP SURVEY":
                return self.finish_survey(state, skip_survey=True)

        # Append user message
        if isinstance(state.full_history[-1], HumanMessage):
            state.narrative.survey_conversation.append(state.full_history[-1])

        # Generate next survey question
        next_question = self.conduct_survey(state.narrative.survey_conversation)
        next_question = self.add_agent_metadata(next_question)

        # Append AI turn
        state.narrative.survey_conversation.append(next_question)

        # Update full_history and last_agent
        state.full_history.append(next_question)
        state.last_agent = self.name

        # Check if the survey is finished
        if next_question.content.endswith("<END>"):
            state = self.finish_survey(state)

        return state

    def conduct_survey(self, survey_conversation):
        print(f"\n--- Asking Survey Question ---")

        # Append system prompt
        messages = [SystemMessage(content=self.survey_prompt)] + survey_conversation

        next_question = self.model.invoke(messages)
        print(f"\nnext_question: {next_question!r}", end="\n\n")

        return next_question

    def finish_survey(self, state: FullState, skip_survey: bool = False):
        """
        Finish the survey and prepare the story to begin.
        """
        print("[NarrativeAgent] Finishing survey...")

        # Set finished_survey to True
        state.narrative.finished_survey = True

        # If skip_survey is True, use default survey results
        if skip_survey:
            print("[NarrativeAgent] Skipping survey and using default survey results.")
            state.narrative.survey_data = str(default_survey_results)
        else:
            # Format the survey results
            state.narrative.survey_data = self.format_survey_results(state.narrative.survey_conversation)

        self.prompt = self.prompt.format(survey_results=state.narrative.survey_data)

        # Reset the story to start fresh
        start_message = AIMessage(content="**--- BEGINNING STORY ---**\n---\n")
        start_message = self.add_agent_metadata(start_message)
        state.narrative.story = [start_message]

        # Update full history and last agent
        state.full_history.append(start_message)
        state.last_agent = self.name

        return state

    def generate_story_segment(self, current_narrative, challenge_prompt=None, next_challenge=None):
        """
        Generate a story segment based on the current narrative and (optionally) a challenge prompt.
        """
        print(f"\n--- Generating Story Segment ---")

        # Append system prompt
        if challenge_prompt:
            prompt = challenge_prompt
        else:
            prompt = self.prompt
        messages = [SystemMessage(content=prompt)] + current_narrative

        # print(f"\nmessages: {messages!r}", end="\n\n")

        story_segment = self.model.invoke(messages)
        # print(f"\nGenerated story segment: {story_segment.content}", end="\n\n")

        print(f"\n[Narrative] challenge_prompt: {challenge_prompt}\n")
        print(f"\n[Narrative] next_challenge: {next_challenge}\n")

        # If this is a challenge, add the triplet as metadata
        if challenge_prompt and next_challenge:
            if getattr(story_segment, 'metadata', None) is None:
                story_segment.metadata = {}
            story_segment.metadata['challenge_triplet'] = next_challenge
        return story_segment

    def add_agent_metadata(self, message):
        if isinstance(message, AIMessage):
            if getattr(message, 'metadata', None) is None:
                message.metadata = {}
            message.metadata["agent"] = self.name
            return message

    def format_survey_results(self, survey_conversation):
        """
        Format the survey results into a dictionary.
        """
        print(f"\n--- Formatting Survey Results ---")

        # Convert the survey conversation into a string
        survey_conversation_string = ""
        for msg in survey_conversation:
            if isinstance(msg, HumanMessage):
                survey_conversation_string += (f"\nUser: {msg.content!r}")
            elif isinstance(msg, AIMessage):
                survey_conversation_string += (f"\nAI: {msg.content!r}")

        # Append system prompt
        messages = [SystemMessage(content=self.survey_format_prompt), HumanMessage(content=f"Here is the conversation for you to turn into a dictionary of survey data collected on the user:\n\n{survey_conversation_string}")]

        survey_data = self.model.invoke(messages)
        print(f"\nSurvey data: {survey_data.content}", end="\n\n")

        return survey_data.content.strip()
