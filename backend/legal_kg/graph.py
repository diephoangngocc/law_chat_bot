from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from .text import compact


ARTICLE_RE = re.compile(r"(?:Điều|D)(\d+)", re.IGNORECASE)


@dataclass(slots=True)
class Node:
    id: str
    name: str
    label: str
    source_file: str = ""


@dataclass(slots=True)
class Edge:
    source: str
    target: str
    relationship: str
    source_file: str = ""


@dataclass(slots=True)
class PointEvidence:
    point_id: str
    point_name: str
    parent_logic_id: str = ""
    parent_logic_name: str = ""
    graph_paths: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        parts = []
        if self.parent_logic_name:
            parts.append(self.parent_logic_name)
        parts.append(self.point_name)
        return "\n".join(parts)


@dataclass(slots=True)
class ClauseEvidence:
    clause_id: str
    clause_name: str
    actions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    points: list[PointEvidence] = field(default_factory=list)
    logic_nodes: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    graph_paths: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        parts = [self.clause_name]
        if self.actions:
            parts.append("Hành vi: " + "; ".join(self.actions))
        if self.conditions:
            parts.append("Điều kiện/điểm: " + "; ".join(self.conditions))
        if self.penalties:
            parts.append("Hình phạt: " + "; ".join(self.penalties))
        return "\n".join(parts)


@dataclass(slots=True)
class ArticleEvidence:
    article_id: str
    article_no: int | None
    title: str
    chapter: str = ""
    clauses: list[ClauseEvidence] = field(default_factory=list)

    def as_text(self) -> str:
        parts = [self.chapter, self.title] if self.chapter else [self.title]
        parts.extend(clause.as_text() for clause in self.clauses)
        return "\n".join(p for p in parts if p)

    def brief(self, max_chars: int = 2200) -> str:
        return compact(self.as_text(), max_chars=max_chars)


class LegalKnowledgeGraph:
    def __init__(self, nodes: dict[str, Node], edges: list[Edge]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.out_edges: dict[str, list[Edge]] = defaultdict(list)
        self.in_edges: dict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            self.out_edges[edge.source].append(edge)
            self.in_edges[edge.target].append(edge)

    @classmethod
    def from_csv_dir(cls, data_dir: str | Path) -> "LegalKnowledgeGraph":
        data_dir = Path(data_dir)
        nodes: dict[str, Node] = {}
        edges: list[Edge] = []

        for path in sorted(data_dir.glob("node*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    node_id = (row.get("ID") or "").strip()
                    if not node_id:
                        continue
                    nodes[node_id] = Node(
                        id=node_id,
                        name=(row.get("Name") or "").strip(),
                        label=(row.get("Label") or "").strip(),
                        source_file=path.name,
                    )

        for path in sorted(data_dir.glob("edge*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    source = (row.get("From") or "").strip()
                    target = (row.get("To") or "").strip()
                    relationship = (row.get("Relationship") or "").strip()
                    if source and target:
                        edges.append(Edge(source, target, relationship, path.name))

        graph = cls(nodes, edges)
        graph.validate()
        return graph

    def validate(self) -> None:
        missing = [
            edge
            for edge in self.edges
            if edge.source not in self.nodes or edge.target not in self.nodes
        ]
        if missing:
            sample = ", ".join(f"{e.source}->{e.target}" for e in missing[:5])
            raise ValueError(f"KG contains edges with missing nodes: {sample}")

    def children(self, node_id: str, relationship: str | None = None) -> list[Node]:
        result = []
        for edge in self.out_edges.get(node_id, []):
            if relationship is None or edge.relationship == relationship:
                target = self.nodes.get(edge.target)
                if target:
                    result.append(target)
        return result

    def parents(self, node_id: str, relationship: str | None = None) -> list[Node]:
        result = []
        for edge in self.in_edges.get(node_id, []):
            if relationship is None or edge.relationship == relationship:
                source = self.nodes.get(edge.source)
                if source:
                    result.append(source)
        return result

    def article_nodes(self) -> list[Node]:
        return sorted(
            (node for node in self.nodes.values() if node.label == "Điều"),
            key=lambda node: (extract_article_no(node) or 10_000, node.id),
        )

    def build_article_evidence(self) -> list[ArticleEvidence]:
        return [self._build_one_article(node) for node in self.article_nodes()]

    def _build_one_article(self, article_node: Node) -> ArticleEvidence:
        chapter = ""
        for parent in self.parents(article_node.id, "Gồm"):
            if parent.label == "Chương":
                chapter = parent.name
                break

        clauses: list[ClauseEvidence] = []
        for child in self.children(article_node.id, "Gồm"):
            if child.label != "Khoản":
                continue
            clauses.append(self._build_clause_evidence(child))

        return ArticleEvidence(
            article_id=article_node.id,
            article_no=extract_article_no(article_node),
            title=article_node.name,
            chapter=chapter,
            clauses=clauses,
        )

    def _build_clause_evidence(self, clause_node: Node) -> ClauseEvidence:
        evidence = ClauseEvidence(clause_id=clause_node.id, clause_name=clause_node.name)

        for edge in self.out_edges.get(clause_node.id, []):
            target = self.nodes[edge.target]
            if edge.relationship == "Quy định":
                evidence.actions.append(target.name)
                evidence.graph_paths.append(path_text([clause_node, target], edge.relationship))
            elif edge.relationship == "Áp dụng":
                evidence.penalties.append(target.name)
                evidence.graph_paths.append(path_text([clause_node, target], edge.relationship))
            elif edge.relationship == "Cần điều kiện":
                evidence.logic_nodes.append(target.name)
                evidence.graph_paths.append(path_text([clause_node, target], edge.relationship))
                self._collect_logic_subtree(target, evidence)
            elif edge.relationship == "Gồm" and target.label in {"Điểm", "Node Logic OR", "Node Logic AND"}:
                evidence.conditions.append(target.name)
                evidence.graph_paths.append(path_text([clause_node, target], edge.relationship))
                if target.label == "Điểm":
                    add_point_evidence(
                        evidence,
                        point_node=target,
                        parent_logic=None,
                        graph_path=path_text([clause_node, target], edge.relationship),
                    )

        return evidence

    def _collect_logic_subtree(self, logic_node: Node, evidence: ClauseEvidence) -> None:
        queue: deque[tuple[Node, list[Node]]] = deque([(logic_node, [logic_node])])
        seen = {logic_node.id}
        while queue:
            current, path = queue.popleft()
            for edge in self.out_edges.get(current.id, []):
                target = self.nodes[edge.target]
                if target.id in seen:
                    continue
                seen.add(target.id)
                next_path = path + [target]
                evidence.graph_paths.append(path_text(next_path, edge.relationship))
                if edge.relationship == "Áp dụng" or target.label == "Hình phạt":
                    evidence.penalties.append(target.name)
                elif target.label == "Điểm":
                    evidence.conditions.append(target.name)
                    add_point_evidence(
                        evidence,
                        point_node=target,
                        parent_logic=current if current.label.startswith("Node Logic") else None,
                        graph_path=path_text(next_path, edge.relationship),
                    )
                elif target.label.startswith("Node Logic"):
                    evidence.logic_nodes.append(target.name)
                    queue.append((target, next_path))


def extract_article_no(node: Node) -> int | None:
    match = ARTICLE_RE.search(node.name) or ARTICLE_RE.search(node.id)
    return int(match.group(1)) if match else None


def add_point_evidence(
    evidence: ClauseEvidence,
    point_node: Node,
    parent_logic: Node | None,
    graph_path: str,
) -> None:
    if all(point.point_id != point_node.id for point in evidence.points):
        evidence.points.append(
            PointEvidence(
                point_id=point_node.id,
                point_name=point_node.name,
                parent_logic_id=parent_logic.id if parent_logic else "",
                parent_logic_name=parent_logic.name if parent_logic else "",
                graph_paths=[graph_path] if graph_path else [],
            )
        )
    if point_node.name not in evidence.conditions:
        evidence.conditions.append(point_node.name)


def path_text(nodes: list[Node], relationship: str) -> str:
    if len(nodes) < 2:
        return nodes[0].name if nodes else ""
    return f"{nodes[0].name} -[{relationship}]-> {nodes[-1].name}"
