from langgraph.graph import StateGraph, START, END

from src.agents.state import ResearchState
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.verifier import verifier_node
from src.agents.synthesizer import synthesizer_node


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "verifier")
    graph.add_edge("verifier", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()