import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from main import app, survey_results, config

def respond(user_message, history):
    # history: List of (user_str, bot_str)
    if history is None:
        history = []
    # rebuild BaseMessage list from history
    messages = []
    for u, b in history:
        messages.append(HumanMessage(u))
        messages.append(AIMessage(b))
    # append new user message
    messages.append(HumanMessage(user_message))

    # prepare input data with your survey defaults
    input_data = survey_results.copy()
    input_data["messages"] = messages

    # invoke the workflow
    output = app.invoke(input_data, config)
    ai_msg = output["messages"][-1]

    # update history for display
    history.append((user_message, ai_msg.content))
    return history, ""  # clear the textbox

with gr.Blocks() as demo:
    gr.Markdown("## 🧙‍♂️ Child-Friendly Storyteller Chat")
    chatbot = gr.Chatbot(elem_id="chatbot")
    txt = gr.Textbox(
        placeholder="Type a message and hit enter",
        show_label=False
    )
    txt.submit(respond, [txt, chatbot], [chatbot, txt])
    gr.Button("Clear").click(lambda: ([], ""), None, [chatbot, txt])

if __name__ == "__main__":
    demo.launch()
