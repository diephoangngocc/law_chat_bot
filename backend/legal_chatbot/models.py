from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    id: str
    name: str
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        values = [self.id, self.name, self.label]
        for key, value in self.metadata.items():
            if value is not None:
                values.append(str(value))
        return " | ".join(v for v in values if v)


@dataclass
class Edge:
    source: str
    target: str
    relation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    id: str
    title: str
    content: str
    source: str
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    document: Document
    score: float
    score_parts: Dict[str, float] = field(default_factory=dict)
    kg_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.document.id,
            "title": self.document.title,
            "content": self.document.content,
            "source": self.document.source,
            "node_id": self.document.node_id,
            "score": round(float(self.score), 4),
            "score_parts": {k: round(float(v), 4) for k, v in self.score_parts.items()},
            "kg_paths": self.kg_paths[:5],
            "metadata": self.document.metadata,
        }
