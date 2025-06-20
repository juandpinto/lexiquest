import gradio as gr
import os
import uuid
import traceback
from typing import Optional, List, Dict, Any

# Langchain and LLM imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import from the new graph location
from core.graph import initialize_graph # MODIFIED IMPORT

# Define a configurable thread_id
def generate_thread_id(prefix: str = "", length: int = 8) -> str:
    """
    Generates a unique thread ID with an optional prefix and specified length.
    Default length is 8 characters.
    """
    return f'LQ-{prefix}_{str(uuid.uuid4())[:length]}'

# Global state variables
LANG_GRAPH_APP: Optional[Any] = None
CURRENT_LLM_INFO: str = "LLM: Not yet determined. Press 'Let's start!' or send a message."
CURRENT_THREAD_ID: str = generate_thread_id(prefix="chat")
PREVIOUS_API_KEY_USED: Optional[str] = None


# LLM selection logic
def get_llm(api_key_from_ui: Optional[str] = None):
    """
    Selects and initializes an LLM based on API key availability and type.
    Priority:
    1. API key from UI (detects OpenAI vs Google based on prefix).
    2. OPENAI_API_KEY from environment.
    3. GOOGLE_API_KEY from environment.
    4. Ollama as a fallback.
    Returns:
        tuple: (llm_instance, llm_info_string)
    Raises:
        ValueError: If no LLM can be initialized.
    """
    # Attempt with UI key first
    if api_key_from_ui:
        if api_key_from_ui.startswith("sk-"):
            print("Attempting to use OpenAI LLM with API key from UI.")
            try:
                llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key_from_ui, temperature=0.8)
                return llm, "OpenAI (GPT-3.5-Turbo via UI)"
            except Exception as e:
                print(f"Error initializing OpenAI with UI key: {e}. Falling back to environment variables or Ollama.")
        elif api_key_from_ui.startswith("AIza"): # Google API keys typically start with "AIza"
            print("Attempting to use Google LLM with API key from UI.")
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key_from_ui, temperature=0.8) # Changed to gemini-2.0-flash, ensure model name is correct
                return llm, "Google (Gemini 2.0 Flash via UI)"
            except Exception as e:
                print(f"Error initializing Google with UI key: {e}. Falling back to environment variables or Ollama.")
        else:
            print(f"API key from UI has an unrecognized prefix ('{api_key_from_ui[:4]}...'). Will try environment variables or Ollama.")

    openai_api_key_env = os.environ.get("OPENAI_API_KEY")
    if openai_api_key_env:
        print("Attempting to use OpenAI LLM with API key from environment.")
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key_env, temperature=0.8)
            return llm, "OpenAI (GPT-3.5-Turbo via ENV)"
        except Exception as e:
            print(f"Error initializing OpenAI with ENV key: {e}. Falling back to Google ENV or Ollama.")

    google_api_key_env = os.environ.get("GOOGLE_API_KEY")
    if google_api_key_env:
        print("Attempting to use Google LLM with API key from environment.")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key_env, temperature=0.8) # Changed to gemini-2.0-flash
            return llm, "Google (Gemini 2.0 Flash via ENV)"
        except Exception as e:
            print(f"Error initializing Google with ENV key: {e}. Falling back to Ollama.")

    print("Attempting to use Ollama LLM (gemma3) as fallback.")
    try:
        # Ensure gemma3 is a valid model name for your ChatOllama setup
        llm = ChatOllama(model="gemma3", temperature=0.8) # Using a common naming for Ollama models
        return llm, "Ollama (gemma)"
    except Exception as e:
        error_message = f"Error initializing Ollama: {e}. No LLM could be initialized. Please check API keys or ensure Ollama server is running."
        print(error_message)
        raise ValueError(error_message) from e


# Function to initialize/get graph and LLM info
def ensure_graph_initialized(api_key_ui: Optional[str]):
    global LANG_GRAPH_APP, CURRENT_LLM_INFO, PREVIOUS_API_KEY_USED, CURRENT_THREAD_ID

    if LANG_GRAPH_APP is None or PREVIOUS_API_KEY_USED != api_key_ui:
        print(f"Initializing graph. App is None: {LANG_GRAPH_APP is None}. API key changed: {PREVIOUS_API_KEY_USED != api_key_ui if PREVIOUS_API_KEY_USED else 'N/A' != api_key_ui}")
        try:
            llm, llm_info = get_llm(api_key_ui)
            LANG_GRAPH_APP = initialize_graph(llm) # This now calls the function from src.core.graph
            CURRENT_LLM_INFO = f"LLM: {llm_info} (Thread: {CURRENT_THREAD_ID})"
            PREVIOUS_API_KEY_USED = api_key_ui
            print(f"Graph initialized successfully with {llm_info} on thread {CURRENT_THREAD_ID}")
        except Exception as e:
            error_msg = f"Error initializing chat system: {e}"
            print(error_msg)
            LANG_GRAPH_APP = None # Ensure app is None if initialization fails
            CURRENT_LLM_INFO = f"Error: {str(e)}. Check API key or Ollama."
            # Do not raise here; let UI handlers show the error via CURRENT_LLM_INFO
    return LANG_GRAPH_APP, CURRENT_LLM_INFO


# --- Gradio Chat Function ---
def chat_interface_function(message_text: str, api_key_ui: str): # api_key_ui is for ensure_graph_initialized
    global LANG_GRAPH_APP, CURRENT_THREAD_ID, CURRENT_LLM_INFO

    # Ensure the graph is initialized with the current API key
    if LANG_GRAPH_APP is None:
        yield f"Chat system initialization failed. Details: {CURRENT_LLM_INFO.replace('LLM: ', '')}"
        return

    # Ensure the thread ID is set
    langgraph_config = {"configurable": {"thread_id": CURRENT_THREAD_ID}}
    current_turn_input = {"full_history": [HumanMessage(content=message_text)]}

    output = ""
    for chunk, metadata in LANG_GRAPH_APP.stream(
            current_turn_input,
            langgraph_config,
            stream_mode="messages",
    ):
        if metadata['langgraph_node'] == 'narrative_agent':
            output += chunk.content
            yield output
        else:
            yield f"*{metadata['langgraph_node']} is processing...*"


# --- Gradio UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## LexiQuest")
    gr.Markdown("Enter your API Key below. You can use OpenAI or Google API keys.")

    with gr.Row():
        api_key_textbox = gr.Textbox(
            label="API Key",
            type="password",
            placeholder="sk-... or AIza...",
            lines=1,
            scale=3
        )
        llm_status_display = gr.Markdown(value=CURRENT_LLM_INFO) # Use initial value

    chatbot = gr.Chatbot(
        label="Story Adventure",
        height=400,
        type="messages"
    )
    # --- UI elements with dynamic visibility ---
    with gr.Row():
        msg_textbox = gr.Textbox(
            label="Your reply:",
            placeholder="Type what you want to say or do...",
            lines=1,  # Change to single line for Enter-to-submit
            scale=6,
            visible=False  # Initially hidden
        )
        with gr.Column(scale=2):
            submit_button = gr.Button("Submit", variant="primary", visible=False)  # Initially hidden
            clear_button = gr.Button("Start over", visible=False)  # Initially hidden
    start_button = gr.Button("Let's start!", variant="primary", visible=True)  # Initially visible

    # Event handler for API key changes
    def handle_api_key_change_effect(api_key: str):
        _app, llm_info_status = ensure_graph_initialized(api_key)
        return llm_info_status

    api_key_textbox.change(
        fn=handle_api_key_change_effect,
        inputs=[api_key_textbox],
        outputs=[llm_status_display],
        show_progress="hidden"
    )

    # Function to handle starting the story
    def handle_start_story(current_display_history: List[Dict[str, Any]], api_key: str):
        _app, llm_info_status = ensure_graph_initialized(api_key)
        if not isinstance(current_display_history, list): current_display_history = []
        full_display_history = current_display_history + [{"role": "assistant", "content": ""}]
        if not _app:
            full_display_history[-1]["content"] = f"Failed to start: {llm_info_status.replace('LLM: ', '').replace('Error: ', '')}"
            # Show start button, hide others
            yield full_display_history, llm_info_status, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return
        # Show chat controls, hide start button
        yield full_display_history, llm_info_status, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        full_response = ""
        for ai_response_chunk in chat_interface_function("Let's start!", api_key):
            full_response = ai_response_chunk
            full_display_history[-1]["content"] = full_response
            yield full_display_history, llm_info_status, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    # Function to handle message submission
    def handle_submit(message_text: str, current_display_history: List[Dict[str, Any]], api_key: str):
        if not message_text.strip():
            yield current_display_history, CURRENT_LLM_INFO, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            return
        _app, llm_info_status = ensure_graph_initialized(api_key)
        if not isinstance(current_display_history, list): current_display_history = []
        full_display_history = current_display_history + [{"role": "user", "content": message_text}]
        full_display_history = full_display_history + [{"role": "assistant", "content": ""}]
        if not _app:
            full_display_history[-1]["content"] = f"Failed to process: {llm_info_status.replace('LLM: ', '').replace('Error: ', '')}"
            yield full_display_history, llm_info_status, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            return
        full_display_history[-1]["content"] = "..."
        yield full_display_history, llm_info_status, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        full_response = ""
        for ai_response_chunk in chat_interface_function(message_text, api_key):
            full_response = ai_response_chunk
            full_display_history[-1]["content"] = full_response
            yield full_display_history, llm_info_status, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    # Function to clear chat and reset thread
    def clear_chat_and_reset_thread():
        global LANG_GRAPH_APP, CURRENT_THREAD_ID, CURRENT_LLM_INFO, PREVIOUS_API_KEY_USED
        LANG_GRAPH_APP = None
        PREVIOUS_API_KEY_USED = None
        CURRENT_THREAD_ID = generate_thread_id(prefix="chat")
        new_llm_status = f"Chat cleared. New Thread: {CURRENT_THREAD_ID}. LLM will re-initialize on next message."
        CURRENT_LLM_INFO = new_llm_status
        print(f"Chat cleared. New thread: {CURRENT_THREAD_ID}. Graph will re-initialize on next interaction.")
        # Hide chat controls, show start button
        return [], new_llm_status, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    # --- Button and event wiring ---
    start_button.click(
        fn=handle_start_story,
        inputs=[chatbot, api_key_textbox],
        outputs=[chatbot, llm_status_display, start_button, msg_textbox, clear_button, submit_button],
        show_progress=True
    )
    submit_button.click(
        fn=handle_submit,
        inputs=[msg_textbox, chatbot, api_key_textbox],
        outputs=[chatbot, llm_status_display, start_button, msg_textbox, clear_button, submit_button],
        show_progress=True
    )
    msg_textbox.submit(
        fn=handle_submit,
        inputs=[msg_textbox, chatbot, api_key_textbox],
        outputs=[chatbot, llm_status_display, start_button, msg_textbox, clear_button, submit_button],
        show_progress=True
    )
    msg_textbox.submit(lambda: gr.update(value=""), inputs=[], outputs=[msg_textbox])
    clear_button.click(
        fn=clear_chat_and_reset_thread,
        inputs=[],
        outputs=[chatbot, llm_status_display, start_button, msg_textbox, clear_button, submit_button],
        show_progress=True
    )


# --- Main Execution ---
if __name__ == "__main__":
    # Load environment variables if using python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Attempting to load .env file for API keys (if configured via python-dotenv).")
    except ImportError:
        print("python-dotenv not installed, skipping .env file loading.")

    print(f"OpenAI API Key from env: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not set'}")
    print(f"Google API Key from env: {'Set' if os.environ.get('GOOGLE_API_KEY') else 'Not set'}")
    print(f"Initial Thread ID: {CURRENT_THREAD_ID}")

    demo.launch()
