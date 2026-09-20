import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():
    """
    Shared Groq LLM configuration.

    The model is intentionally configured with bounded retries so transient
    rate-limit/server failures do not immediately fail the whole workflow.
    Structured JSON response mode is NOT enabled here because the Groq model
    can reject otherwise valid requests with json_validate_failed.
    """

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        max_tokens=700,
        max_retries=2,
    )