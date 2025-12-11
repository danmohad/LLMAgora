"""Core package for the Agora arena."""

from dotenv import load_dotenv

# Load environment variables (e.g., OPENROUTER_API_KEY) once package is imported.
load_dotenv()

__all__ = ["agent", "agora", "llm", "memory", "persistence", "workflows"]
