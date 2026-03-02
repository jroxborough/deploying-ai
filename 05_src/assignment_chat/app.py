"""
app.py
──────
Gradio chat interface for Barnaby's Bookshop.
Run with: python app.py
"""

import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

from agent import chat, reset_conversation
from config import PERSONA_NAME, SHOP_NAME

# ── Barnaby's personality intro ───────────────────────────────────────────────

WELCOME_MESSAGE = f"""Welcome to *{SHOP_NAME}*! 👩🏻‍🦰📚

I'm {PERSONA_NAME} — I've been running this little shop for thirty years, and I've read (or at least argued about) most of what's on these shelves. 

Here's what I can help you with:
- **Look up a specific book or author** — just ask me about it by name
- **Get personalised recommendations** — describe a mood, theme, or vibe and I'll find something
- **Manage your reading list** — tell me to add, remove, or show your list
- **Filter by criteria** — short reads, specific genres, highly rated, you name it

So then — what are you in the mood for?"""


# ── Chat handler ──────────────────────────────────────────────────────────────

def respond(message: str, history: list) -> str:
    """Process a user message and return the assistant's response."""
    if not message.strip():
        return ""
    try:
        return chat(message)
    except Exception as e:
        return f"My apologies — something went wrong in the back room. ({str(e)})"


def clear_chat():
    """Reset conversation memory when the user clears the chat."""
    reset_conversation()
    return []


# ── Gradio Interface ──────────────────────────────────────────────────────────

with gr.Blocks(
    title=SHOP_NAME,
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="orange",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("Lora"), "Georgia", "serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ),
    css="""
    .gradio-container {
        max-width: 800px !important;
        margin: 0 auto;
    }
    .barnaby-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        border-bottom: 2px solid #d97706;
        margin-bottom: 1rem;
    }
    .barnaby-header h1 {
        font-family: 'Lora', Georgia, serif;
        font-size: 2rem;
        color: #92400e;
        margin: 0;
    }
    .barnaby-header p {
        color: #78716c;
        font-style: italic;
        margin: 0.25rem 0 0;
        font-size: 0.95rem;
    }
    footer { display: none !important; }
    """
) as demo:

    gr.HTML(f"""
    <div class="barnaby-header">
        <h1>📚 {SHOP_NAME}</h1>
        <p>Est. 1996 · Independent · Opinionated · Well-Read</p>
    </div>
    """)

    chatbot = gr.Chatbot(
        value=[[None, WELCOME_MESSAGE]],
        label="",
        height=520,
        show_label=False,
        avatar_images=(
            None,
            f"https://api.dicebear.com/7.x/bottts/svg?seed={PERSONA_NAME.lower()}&backgroundColor=b45309"
        ),
        bubble_full_width=False,
        render_markdown=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about a book, describe what you're in the mood for, or manage your reading list...",
            show_label=False,
            scale=9,
            container=False,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    with gr.Row():
        clear_btn = gr.Button("🗑️ New Conversation", variant="secondary", size="sm")
        gr.Markdown(
            "*Powered by Open Library API · CMU Book Summaries · OpenAI*",
            elem_classes=["footer-note"]
        )

    # Example prompts
    gr.Examples(
        examples=[
            "Tell me about The Great Gatsby by F. Scott Fitzgerald",
            "I want something dark and atmospheric, set in the past",
            "Add The Remains of the Day to my reading list",
            "Show me my reading list",
            "Find me something under 250 pages that isn't too heavy",
            "Recommend something similar to Never Let Me Go",
        ],
        inputs=msg,
        label="Try asking...",
    )

    # Event handlers
    def user_submit(message, history):
        if not message.strip():
            return "", history
        history = history or []
        history.append([message, None])
        return "", history

    def bot_respond(history):
        if not history or history[-1][1] is not None:
            return history
        user_message = history[-1][0]
        response = respond(user_message, history[:-1])
        history[-1][1] = response
        return history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_respond, chatbot, chatbot
    )
    send_btn.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_respond, chatbot, chatbot
    )
    clear_btn.click(clear_chat, outputs=chatbot)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )