"""Gradio demo for your capstone.  Run:  uv run python capstone/app.py"""

import gradio as gr

from pipeline import TicketAssistant

assistant = TicketAssistant(use_llm=False)  # set True to draft grounded replies with the LLM


def run(ticket: str):
    out = assistant(ticket)
    return out["intent"] or "(abstained → human review)", out["confidence"], out["priority"], out.get("reply") or "", out


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=4, label="Customer message"),
    outputs=[
        gr.Textbox(label="Intent"),
        gr.Number(label="Confidence"),
        gr.Textbox(label="Priority"),
        gr.Textbox(label="Draft reply", lines=6),
        gr.JSON(label="Full output"),
    ],
    examples=[["Where is my refund? It's been two weeks."], ["لم يصلني الطلب A-48213 حتى الآن"], ["I want to delete my account"]],
    title="Ticket Assistant",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
