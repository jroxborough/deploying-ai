"""
LLM Client
──────────
Shared OpenAI client factory. All LLM calls in the project go through here.
Uses an AWS API Gateway proxy with a custom x-api-key header.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

base_dir = os.path.dirname(os.path.dirname(__file__))

load_dotenv(dotenv_path=os.path.join(base_dir, "..", ".env"))
load_dotenv(dotenv_path=os.path.join(base_dir, "..", ".secrets"))

BASE_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"

def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        base_url=BASE_URL,
        api_key="any value",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
    )

# Testing the connection to the API Gateway and OpenAI
if __name__ == "__main__":
    print("Testing connection...")
    print(f"  Model:   {os.getenv('OPENAI_MODEL')}")
    print(f"  API key: {'SET' if os.getenv('API_GATEWAY_KEY') else 'MISSING'}")

    try:
        llm = get_llm(temperature=0)
        response = llm.invoke("Say hello in one short sentence.")
        print(f"  Response: {response.content}")
        print("✓ Connection successful")
    except Exception as e:
        print(f"✗ Connection failed: {e}")