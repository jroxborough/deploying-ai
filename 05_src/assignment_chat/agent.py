"""
LangGraph Agent
Routes user messages to the appropriate service and manages conversation flow.
"""

import os
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from services.api_service import lookup_book
from services.semantic_service import search_books
from services.tools_service import run_tool_agent
from services.guardrails import run_guardrails
from services.llm_client import get_llm
from memory.memory_manager import MemoryManager
from config import PERSONA, PROMPT_CHITCHAT

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def debug(msg: str):
    if DEBUG:
        print(f"  [AGENT] {msg}")


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    route: str
    response: str


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are a routing assistant for a bookshop chatbot. 
Classify the user's message into exactly one of these categories:

- api_lookup: User wants information about a SPECIFIC known book or author by name
  Examples: "tell me about 1984", "who wrote Dune?", "what is Crime and Punishment about?"

- semantic_search: User wants recommendations or is describing what they want vaguely
  Examples: "I want something cozy", "a dark thriller set in Japan", "books like Harry Potter", 
  "something thought-provoking but not too long", "what should I read next?"

- tool_call: User wants to manage their reading list or filter by specific criteria
  Examples: "add that to my list", "show my reading list", "find me something under 200 pages",
  "remove Gone Girl", "find books similar to The Road", "I want a highly rated mystery"

- chitchat: General conversation not about a specific book task
  Examples: "hello", "thanks", "what can you do?", "you're great"

Respond with ONLY the category name, nothing else."""


def route_message(state: AgentState) -> AgentState:
    """Classify the user's intent and set the route."""
    llm = get_llm(temperature=0)

    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=state["user_input"])
    ])

    route = response.content.strip().lower()
    if route not in ["api_lookup", "semantic_search", "tool_call", "chitchat"]:
        debug(f"Router returned unexpected value '{route}', defaulting to semantic_search")
        route = "semantic_search"

    debug(f"Router decided: '{route}' for input: '{state['user_input'][:60]}'")
    return {**state, "route": route}


def decide_next_node(state: AgentState) -> str:
    """Conditional edge: route to the correct service node."""
    return state["route"]


# ── Guardrail Node ────────────────────────────────────────────────────────────

def guardrail_node(state: AgentState) -> AgentState:
    debug(f"Guardrail checking: '{state['user_input'][:60]}'")
    blocked, refusal = run_guardrails(state["user_input"])
    if blocked:
        debug("Guardrail BLOCKED message")
        return {**state, "route": "blocked", "response": refusal}
    debug("Guardrail passed")
    return state


def decide_after_guardrail(state: AgentState) -> str:
    """Conditional edge after guardrail: blocked messages skip the router."""
    if state["route"] == "blocked":
        return "blocked"
    return "router"


# ── Service Nodes ─────────────────────────────────────────────────────────────

def api_lookup_node(state: AgentState) -> AgentState:
    """Service 1: Look up a specific book via Open Library API."""
    debug("Calling Service 1: Open Library API")
    response = lookup_book(state["user_input"])
    debug(f"Service 1 response length: {len(response)} chars")
    return {**state, "response": response}


def semantic_search_node(state: AgentState) -> AgentState:
    """Service 2: Semantic search over ChromaDB book summaries."""
    debug("Calling Service 2: Semantic search")
    response = search_books(state["user_input"])
    debug(f"Service 2 response length: {len(response)} chars")
    return {**state, "response": response}


def tool_call_node(state: AgentState) -> AgentState:
    """Service 3: Function calling for reading list and filters."""
    debug("Calling Service 3: Function calling tools")
    history = state.get("messages", [])[-8:]
    response = run_tool_agent(state["user_input"], history)
    debug(f"Service 3 response length: {len(response)} chars")
    return {**state, "response": response}


def chitchat_node(state: AgentState) -> AgentState:
    """Handle general conversation with Barnaby's personality."""
    llm = get_llm(temperature=0.8)

    recent_messages = state.get("messages", [])[-6:]

    messages = [
        SystemMessage(content=f"{PERSONA}\n\n{PROMPT_CHITCHAT}"),
        *recent_messages,
        HumanMessage(content=state["user_input"])
    ]

    response = llm.invoke(messages)
    return {**state, "response": response.content}


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_agent():
    """Assemble the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("guardrail", guardrail_node)   # ← first stop for every message
    graph.add_node("blocked", lambda s: s)         # passthrough for blocked messages
    graph.add_node("router", route_message)
    graph.add_node("api_lookup", api_lookup_node)
    graph.add_node("semantic_search", semantic_search_node)
    graph.add_node("tool_call", tool_call_node)
    graph.add_node("chitchat", chitchat_node)

    # Entry point is now the guardrail
    graph.set_entry_point("guardrail")

    # After guardrail: blocked messages go to END, safe messages go to router
    graph.add_conditional_edges(
        "guardrail",
        decide_after_guardrail,
        {
            "blocked": "blocked",
            "router": "router",
        }
    )
    graph.add_edge("blocked", END)

    # Conditional routing from router
    graph.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "api_lookup": "api_lookup",
            "semantic_search": "semantic_search",
            "tool_call": "tool_call",
            "chitchat": "chitchat",
        }
    )

    # All service nodes go to END
    for node in ["api_lookup", "semantic_search", "tool_call", "chitchat"]:
        graph.add_edge(node, END)

    return graph.compile()


# ── Singleton Agent + Memory ──────────────────────────────────────────────────

_agent = None
_memory = MemoryManager()


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def chat(user_input: str) -> str:
    """
    Main entry point: process a user message through the agent.
    Manages memory and returns the assistant's response.
    """
    agent = get_agent()

    debug(f"── New message ──────────────────────────────")
    debug(f"Input: '{user_input[:80]}'")
    debug(f"Memory: {_memory.message_count} messages in window")

    _memory.add_user_message(user_input)

    state = AgentState(
        messages=_memory.get_messages_for_llm(),
        user_input=user_input,
        route="",
        response=""
    )

    result = agent.invoke(state)
    response = result["response"]

    debug(f"Final route: '{result['route']}'")
    debug(f"Response preview: '{response[:80]}'")

    _memory.add_ai_message(response)

    return response


def reset_conversation():
    """Clear conversation memory (e.g. for a new session)."""
    _memory.clear()