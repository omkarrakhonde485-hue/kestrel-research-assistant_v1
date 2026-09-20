import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.planner import planner_node


state = {
    "question": "How does the Beacon scheduler handle missed evaluations?",
    "conversation_history": [],
}

result = planner_node(state)

print("\nSearch plan:")
for query in result["search_plan"]:
    print("-", query)