from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.legal_chatbot.config import Settings  # noqa: E402
from backend.legal_chatbot.graph import LegalKnowledgeGraph  # noqa: E402


if __name__ == "__main__":
    settings = Settings.from_env()
    graph = LegalKnowledgeGraph(settings.data_dir)
    print("DATA_DIR:", settings.data_dir)
    print("Graph stats:", graph.stats())
    if not graph.nodes:
        print("Cảnh báo: chưa có node nào. Hãy copy CSV vào data/nodes.")
    if not graph.edges:
        print("Cảnh báo: chưa có edge nào. Hãy copy CSV vào data/edges.")
    print("Một số node đầu tiên:")
    for node in list(graph.nodes.values())[:10]:
        print(f"- {node.id} | {node.label} | {node.name}")
