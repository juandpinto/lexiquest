import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from main import app, survey_results, config

def alternatingly_agree(message, history):
    input_data = survey_results.copy()
    input_data['messages'] = [HumanMessage(message)]

    # output = app.invoke(
    #     input_data,
    #     config
    # )

    output = ""
    for chunk, metadata in app.stream(
        input_data,
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            output += chunk.content
            yield output

    # return output['messages'][-1].content


gr.ChatInterface(
    fn=alternatingly_agree,
    type="messages"
).launch()
