import gradio as gr
import os
import getpass
from typing import Sequence, TypedDict, Optional

# Attempt to load environment variables from .env file
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # Fallback if .env is not in os.getcwd() but maybe one level up or in script dir
        script_dir_dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(script_dir_dotenv_path):
            load_dotenv(dotenv_path=script_dir_dotenv_path)
        else:
            # Try loading without explicit path if python-dotenv is installed
            load_dotenv()

except ImportError:
    print("python-dotenv not installed, .env file will not be loaded automatically.")
    pass

# LangSmith tracing (optional)
os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING", "true")
if "LANGSMITH_API_KEY" not in os.environ and os.environ["LANGSMITH_TRACING"] == "true":
    print("LangSmith API key not found in environment. Tracing might not work.")
if "LANGSMITH_PROJECT" not in os.environ and os.environ["LANGSMITH_TRACING"] == "true":
    print("LangSmith project name not found in environment. Tracing might not work.")


# Langchain and LLM imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from typing_extensions import Annotated, TypedDict

# --- Configuration and initial data ---

survey_results = {
    'age': 7,
    'interests': ['space', 'dinosaurs'],
    'wants_to_be': 'astronaut',
    'favorite_color': 'blue',
    'favorite_food': 'pizza',
    'favorite_animal': 'dolphin',
    'favorite_book': 'Harry Potter',
    'favorite_movie': 'Toy Story',
    'favorite_subject': 'science',
}

prompt_template_str = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a master storyteller who really understands how to engage children. Co-create a simple short story with the child, using short sentences and simple words. The story should be fun, exciting, age-appropriate, and personalized to their interests. The child is {age} years old, wants to be a {wants_to_be} when they grow up, and has the following interests:

            General interests: {interests}.
            Favorite color: {favorite_color}.
            Favorite food: {favorite_food}.
            Favorite animal: {favorite_animal}.
            Favorite book: {favorite_book}.
            Favorite movie: {favorite_movie}.
            Favorite subject: {favorite_subject}.

            Begin by setting the scene, organically asking the child questions to guide the story, and responding to their answers. Use emojis to make it more engaging!""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- LangGraph state definition ---
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    age: int
    interests: Sequence[str]
    wants_to_be: str
    favorite_color: str
    favorite_food: str
    favorite_animal: str
    favorite_book: str
    favorite_movie: str
    favorite_subject: str
    api_key_for_llm: Optional[str]
    selected_llm_info: Optional[str]


# --- LLM Selection Logic ---
def get_llm(api_key_from_ui: Optional[str] = None):
    """Selects and initializes an LLM based on API key availability."""
    openai_api_key = api_key_from_ui or os.environ.get("OPENAI_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    if openai_api_key:
        print("Using OpenAI LLM.")
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.8), "OpenAI (GPT-3.5-Turbo)"
    elif google_api_key:
        print("Using Google LLM.")
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=0.8), "Google (Gemini-Flash)"
    else:
        print("Using Ollama LLM (gemma3).")
        # Ensure Ollama server is running if you use this.
        return ChatOllama(model="gemma3", temperature=0.8), "Ollama (gemma3)"

# --- LangGraph Workflow Definition ---
workflow = StateGraph(state_schema=State)

def call_model_node(state: State):
    """Node that calls the selected LLM."""
    api_key_to_use = state.get("api_key_for_llm")
    llm, llm_info = get_llm(api_key_to_use)

    # Ensure all necessary fields for the prompt are in the state
    prompt_values = {key: state[key] for key in survey_results.keys() if key in state}
    prompt_values["messages"] = state["messages"]

    prompt = prompt_template_str.invoke(prompt_values)
    response = llm.invoke(prompt)
    return {"messages": [response], "selected_llm_info": llm_info}

workflow.add_node("model", call_model_node)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# LangGraph configuration (fixed thread_id for simplicity)
# For multi-user apps, thread_id needs to be managed per user/session.
langgraph_config = {"configurable": {"thread_id": "gradio-chat-thread-1"}}

# --- Gradio Chat Function ---
def chat_interface_function(message_text: str, history_list_of_lists: list, api_key_ui: str):
    """
    Handles chat interaction for the Gradio interface.
    'history_list_of_lists' is from gr.Chatbot: [[user_msg, ai_msg], ...]
    'api_key_ui' is the API key from the Gradio text input.
    """

    current_turn_input = {
        **survey_results,
        "messages": [HumanMessage(content=message_text)],
        "api_key_for_llm": api_key_ui if api_key_ui else None,
    }

    # Stream the response from LangGraph
    try:
        output = ""
        for chunk, metadata in app.stream(
            current_turn_input,
            langgraph_config,
            stream_mode="messages",
        ):

            if isinstance(chunk, AIMessage):
                output += chunk.content
                yield output

    except Exception as e:
        print(f"Error during LangGraph stream: {e}")
        yield f"An error occurred: {str(e)}" #, llm_info_display

# --- Gradio UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Interactive Storyteller Chat")
    gr.Markdown("Enter your API Key below. If left empty, will try to use `OPENAI_API_KEY` or `GOOGLE_API_KEY` from your `.env` file, or fall back to a local Ollama model.")

    with gr.Row():
        api_key_textbox = gr.Textbox(
            label="API Key (Optional, e.g., OpenAI, Google)",
            type="password",
            placeholder="sk-... or AIza...",
            lines=1,
            scale=3
        )
        llm_status_display = gr.Markdown("LLM: Not yet determined. Will update after first message or if API key changes.")


    chatbot = gr.Chatbot(
        label="Story Adventure",
        height=400,
        type="messages"
    )
    msg_textbox = gr.Textbox(
        label="Your reply:",
        placeholder="Type what you want to say or do...",
        lines=2,
        scale=4
    )

    with gr.Row():
        start_button = gr.Button("Let's start!", variant="primary")
        clear_button = gr.Button("Clear Chat & Reset Thread")

    def handle_start_story(history: list, api_key: str):
        # Add current user message to history
        history.append({"role": "user", "content": ""})

        # Add a placeholder for the assistant's response, which will be updated by the stream.
        history.append({"role": "assistant", "content": ""})

        for ai_response_chunk in chat_interface_function("Let's start!", history[:-2], api_key):
            history[-1]["content"] = ai_response_chunk
            yield history

    def handle_submit(message_text: str, history: list, api_key: str):
        history.append({"role": "user", "content": message_text})

        # Add a placeholder for the assistant's response, which will be updated by the stream.
        history.append({"role": "assistant", "content": ""})

        for ai_response_chunk in chat_interface_function(message_text, history[:-2], api_key):
            history[-1]["content"] = ai_response_chunk
            yield history

    start_button.click(
        fn=handle_start_story,
        inputs=[chatbot, api_key_textbox],
        outputs=[chatbot]
    )

    msg_textbox.submit(
        fn=handle_submit,
        inputs=[msg_textbox, chatbot, api_key_textbox],
        outputs=[chatbot]
    )
    msg_textbox.submit(lambda: "", inputs=[], outputs=[msg_textbox])

    def clear_chat_and_reset_thread():
        # Reset LangGraph memory for this thread by changing the thread_id or clearing it.
        # For MemorySaver, clearing involves deleting the checkpoint.
        # A simpler way for this demo is to use a new thread_id, but that needs app re-compilation
        # or more complex config management.
        # For now, we'll just clear the UI. LangGraph memory for "gradio-chat-thread-1" will persist unless manually cleared.
        # To truly reset, you'd need to interact with the checkpointer.
        # A pragmatic approach for a demo: tell the user the backend thread is NOT reset by this UI button alone
        # unless we add logic to clear the specific checkpoint file used by MemorySaver.

        # For this example, let's clear the checkpoint if possible (MemorySaver stores in memory by default,
        # unless configured with a file path, so "clearing" is more about starting a new thread or re-compiling).
        # Since MemorySaver is in-memory by default, re-compiling app or changing thread_id in config
        # would be a way to "reset".

        # The simplest UI reset:
        return [], "Chat cleared. Note: Underlying conversation memory for the fixed thread_id may persist."

    clear_button.click(
        fn=clear_chat_and_reset_thread,
        inputs=[],
        outputs=[chatbot, llm_status_display]
    )


# --- Main Execution ---
if __name__ == "__main__":
    print("Attempting to load .env file for API keys...")
    print(f"OpenAI API Key from env: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not set'}")
    print(f"Google API Key from env: {'Set' if os.environ.get('GOOGLE_API_KEY') else 'Not set'}")
    demo.launch()
