"""
Memory Manager
Handles conversation history with trimming and summarization
for long conversations that would exceed the context window.
"""

import os
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage
)
from services.llm_client import get_llm

# Keep the last N message pairs before summarizing older context
MAX_MESSAGES = 12  # ~6 turns of conversation
SUMMARY_THRESHOLD = 20  # summarize when we exceed this many messages


class MemoryManager:
    def __init__(self):
        self.messages: list[BaseMessage] = []
        self.summary: str = ""  # rolling summary of older context

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
        self._maybe_summarize()

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def get_messages_for_llm(self) -> list[BaseMessage]:
        """
        Return messages to pass to the LLM.
        If there's a summary, prepend it as a system context note.
        """
        if self.summary:
            summary_msg = SystemMessage(
                content=f"[Earlier in this conversation: {self.summary}]"
            )
            return [summary_msg] + self.messages
        return self.messages

    def _maybe_summarize(self):
        """
        If the conversation is getting long, summarize the oldest messages
        and trim them from the active window.
        """
        if len(self.messages) <= SUMMARY_THRESHOLD:
            return

        # Keep the most recent MAX_MESSAGES, summarize the rest
        messages_to_summarize = self.messages[:-MAX_MESSAGES]
        self.messages = self.messages[-MAX_MESSAGES:]

        new_summary = self._summarize(messages_to_summarize)

        # Combine with existing summary if there is one
        if self.summary:
            self.summary = f"{self.summary} Later: {new_summary}"
        else:
            self.summary = new_summary

    def _summarize(self, messages: list[BaseMessage]) -> str:
        """Use the LLM to summarize a chunk of conversation history."""
        llm = get_llm(temperature=0)

        conversation_text = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Barnaby'}: {m.content}"
            for m in messages
        ])

        prompt = f"""Summarize the following bookshop conversation briefly. 
Focus on: books mentioned, user preferences expressed, items added to reading list, 
and any important context for continuing the conversation.
Keep it to 2-3 sentences.

Conversation:
{conversation_text}

Summary:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def clear(self):
        self.messages = []
        self.summary = ""

    @property
    def message_count(self) -> int:
        return len(self.messages)
