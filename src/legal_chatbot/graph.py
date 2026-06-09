from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Document, Edge, Node
from .text import article_number, clause_number, legal_reference_display, lexical_similarity, normalize_text, phrase_count, point_letter, short_text, strip_accents, tokenize, unique_keep_order

NODE_ID_ALIASES = ["id", "node_id", "nodeid", "ma", "mã", "ID", "Id", "NodeID", "node", "Node"]
NODE_NAME_ALIASES = ["name", "ten", "tên", "label_name", "Name", "Ten", "NodeName", "title", "Title", "text", "Text", "content", "Content"]
NODE_LABEL_ALIASES = ["label", "type", "loai", "loại", "Label", "Type", "Loai", "category", "Category"]

EDGE_SOURCE_ALIASES = ["from", "source", "start", "src", "From", "Source", "Start", "start_id", "source_id"]
EDGE_TARGET_ALIASES = ["to", "target", "end", "dst", "To", "Target", "End", "end_id", "target_id"]
EDGE_REL_ALIASES = ["relationship", "relation", "rel", "type", "Relationship", "Relation", "Type", "label", "Label"]

ARTICLE_LABELS = {"điều", "dieu", "article", "law_article"}
PENALTY_HINTS = ["hình phạt", "phạt tù", "phạt tiền", "tử hình", "chung thân", "cải tạo", "khung hình phạt", "mức phạt"]
CLAUSE_HINTS = ["khoản"]
POINT_HINTS = ["điểm"]


def _pick(row: Dict[str, str], aliases: Sequence[str], default: str = "") -> str:
    lower_map = {str(k).strip().lower(): k for k in row.keys()}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lower_map:
            return str(row.get(lower_map[key], default) or "").strip()
    return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
        except Exception:
            return []
    return []


class LegalKnowledgeGraph:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.out_edges: Dict[str, List[Edge]] = defaultdict(list)
        self.in_edges: Dict[str, List[Edge]] = defaultdict(list)
        self._token_index: Dict[str, set[str]] = defaultdict(set)
        self._article_index: Dict[str, List[str]] = defaultdict(list)
        self._name_index: Dict[str, str] = {}
        self.load()

    def load(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.out_edges.clear()
        self.in_edges.clear()
        self._token_index.clear()
        self._article_index.clear()
        self._name_index.clear()

        self._load_nodes(self.data_dir / "nodes")
        self._load_edges(self.data_dir / "edges")
        self._build_indexes()

    def _load_nodes(self, nodes_dir: Path) -> None:
        if not nodes_dir.exists():
            return
        for path in sorted(nodes_dir.glob("*.csv")):
            rows = _read_csv(path)
            for idx, row in enumerate(rows):
                node_id = _pick(row, NODE_ID_ALIASES) or f"{path.stem}:{idx}"
                name = _pick(row, NODE_NAME_ALIASES)
                label = _pick(row, NODE_LABEL_ALIASES)

                if not name:
                    # Fallback: gom toàn bộ row thành text nếu không có cột Name rõ ràng.
                    name = " | ".join(str(v).strip() for v in row.values() if str(v).strip())
                if not label:
                    label = self._infer_label(name, node_id)

                metadata = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    if k.strip().lower() not in {"id", "node_id", "nodeid", "name", "ten", "tên", "label", "type", "loai", "loại"}:
                        metadata[k] = v
                metadata["file"] = path.name
                self.nodes[node_id] = Node(id=node_id, name=name, label=label, metadata=metadata)

    def _load_edges(self, edges_dir: Path) -> None:
        if not edges_dir.exists():
            return
        for path in sorted(edges_dir.glob("*.csv")):
            rows = _read_csv(path)
            for row in rows:
                source = _pick(row, EDGE_SOURCE_ALIASES)
                target = _pick(row, EDGE_TARGET_ALIASES)
                relation = _pick(row, EDGE_REL_ALIASES)
                if not source or not target:
                    continue
                edge = Edge(source=source, target=target, relation=relation, metadata={"file": path.name})
                self.edges.append(edge)
                self.out_edges[source].append(edge)
                self.in_edges[target].append(edge)

    def _infer_label(self, name: str, node_id: str = "") -> str:
        text = normalize_text(f"{node_id} {name}")
        if re.match(r"^điều\s*\d+", text) or re.search(r"\bd(?:ieu)?[_\- ]?\d{1,3}", strip_accents(text)):
            return "Điều"
        if "khoản" in text:
            return "Khoản"
        if "điểm" in text:
            return "Điểm"
        if any(k in text for k in PENALTY_HINTS):
            return "Hình phạt"
        if "hành vi" in text:
            return "Hành vi"
        if any(k in text for k in ["tình tiết", "điều kiện", "cần điều kiện"]):
            return "Tình tiết"
        return ""

    def _build_indexes(self) -> None:
        for node_id, node in self.nodes.items():
            self._name_index[normalize_text(node.name)] = node_id
            art_num = self.article_number_of_node(node)
            if art_num and self.is_article_node(node):
                self._article_index[art_num].append(node_id)

            text_norm = normalize_text(node.text)
            for tok in tokenize(text_norm):
                self._token_index[tok].add(node_id)
            no_acc = strip_accents(text_norm)
            for tok in tokenize(no_acc):
                self._token_index[tok].add(node_id)

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    def to_documents(self) -> List[Document]:
        docs: List[Document] = []
        for node in self.nodes.values():
            docs.append(
                Document(
                    id=f"node:{node.id}",
                    title=node.name or node.id,
                    content=node.text,
                    source="kg_node",
                    node_id=node.id,
                    metadata={"label": node.label, **node.metadata},
                )
            )
        return docs

    def article_number_of_node(self, node: Node) -> Optional[str]:
        return article_number(f"{node.name} {node.label} {node.id}")

    def get_article_nodes(self, number: str) -> List[Node]:
        ids = self._article_index.get(str(number).lower(), [])
        return [self.nodes[i] for i in ids if i in self.nodes]

    def search_nodes(self, query: str, extra_terms: Iterable[str] = (), limit: int = 50, preferred_articles: Iterable[str] = ()) -> List[Tuple[Node, float]]:
        all_terms = unique_keep_order([query, *list(extra_terms)])
        if not all_terms:
            return []

        term_tokens: List[str] = []
        for term in all_terms:
            term_tokens.extend(tokenize(term))
        term_tokens = unique_keep_order(term_tokens)

        scores: Dict[str, float] = defaultdict(float)
        query_norm = normalize_text(" ".join(all_terms))
        query_no_acc = strip_accents(query_norm)

        # Exact article lookup: ưu tiên cực mạnh khi người dùng hỏi Điều 123.
        mentioned_articles = set(preferred_articles or [])
        mentioned_articles.update(article_number(query_norm) for _ in [0] if article_number(query_norm))
        for art in mentioned_articles:
            if not art:
                continue
            for node in self.get_article_nodes(art):
                scores[node.id] += 30.0
                # Thêm hàng xóm trực tiếp của điều luật.
                for e in self.out_edges.get(node.id, []):
                    scores[e.target] += 12.0
                for e in self.in_edges.get(node.id, []):
                    scores[e.source] += 8.0

        # Token index: nhanh và bao quát.
        for term in term_tokens:
            term_no_acc = strip_accents(term)
            for node_id in self._token_index.get(term, set()):
                scores[node_id] += 1.0
            for node_id in self._token_index.get(term_no_acc, set()):
                scores[node_id] += 0.8

        # Phrase boost: quan trọng hơn token với tội danh/cụm pháp lý.
        phrases = [t for t in all_terms if len(tokenize(t)) >= 1]
        for node_id, node in self.nodes.items():
            text_norm = normalize_text(node.text)
            text_no_acc = strip_accents(text_norm)

            if query_norm and query_norm in text_norm:
                scores[node_id] += 10.0
            if query_no_acc and query_no_acc in text_no_acc:
                scores[node_id] += 6.0

            pc = phrase_count(text_norm, phrases)
            if pc:
                scores[node_id] += min(pc * 3.0, 18.0)

            # Soft similarity cho các câu hỏi ngắn như "tội cướp tài sản".
            sim = lexical_similarity(query_norm, text_norm)
            if sim > 0.25:
                scores[node_id] += sim * 4.0

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(self.nodes[node_id], score) for node_id, score in ranked if node_id in self.nodes]

    def expand(self, node_ids: Iterable[str], depth: int = 1, max_nodes: int = 120) -> List[Node]:
        seen = set()
        q = deque((node_id, 0) for node_id in node_ids if node_id in self.nodes)
        out: List[Node] = []
        while q and len(out) < max_nodes:
            node_id, d = q.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            out.append(self.nodes[node_id])
            if d >= depth:
                continue
            for edge in self.out_edges.get(node_id, []):
                if edge.target not in seen:
                    q.append((edge.target, d + 1))
            for edge in self.in_edges.get(node_id, []):
                if edge.source not in seen:
                    q.append((edge.source, d + 1))
        return out

    def is_article_node(self, node: Node) -> bool:
        label = normalize_text(node.label)
        label_no_acc = strip_accents(label)
        name = normalize_text(node.name)
        # Tránh nhầm "Khoản 1 Điều 123" là node Điều.
        if "khoản" in name or "điểm" in name:
            return False
        if label in ARTICLE_LABELS or label_no_acc in ARTICLE_LABELS:
            return True
        return bool(re.match(r"^điều\s*\d+[a-z]?", name)) or bool(re.match(r"^d(?:ieu)?[_\- ]?\d+", strip_accents(normalize_text(node.id))))

    def is_clause_node(self, node: Node) -> bool:
        text = normalize_text(f"{node.label} {node.name}")
        label = normalize_text(node.label)
        # Node Điểm thường chứa cả cụm "Khoản 1 Điều ..." trong tên.
        # Vì vậy phải loại điểm trước, nếu không Điểm sẽ bị nhận nhầm thành Khoản.
        if label == "điểm" or text.startswith("điểm "):
            return False
        return any(k in text for k in CLAUSE_HINTS)

    def is_point_node(self, node: Node) -> bool:
        name = normalize_text(node.name)
        label = normalize_text(node.label)
        # Không nhận nhầm "Hình phạt điểm a" thành node Điểm.
        return label == "điểm" or name.startswith("điểm ")

    def clause_number_of_node(self, node: Node) -> Optional[str]:
        return clause_number(f"{node.name} {node.label} {node.id}")

    def point_letter_of_node(self, node: Node) -> Optional[str]:
        return point_letter(f"{node.name} {node.label} {node.id}")

    def is_penalty_node(self, node: Node) -> bool:
        text = normalize_text(f"{node.label} {node.name} {node.text}")
        return any(p in text for p in PENALTY_HINTS)

    def nearest_article(self, node_id: Optional[str], max_depth: int = 5) -> Optional[Node]:
        if not node_id or node_id not in self.nodes:
            return None
        if self.is_article_node(self.nodes[node_id]):
            return self.nodes[node_id]
        seen = {node_id}
        q = deque([(node_id, 0)])
        while q:
            current, depth = q.popleft()
            if depth >= max_depth:
                continue
            neighbor_ids = []
            neighbor_ids.extend(edge.source for edge in self.in_edges.get(current, []))
            neighbor_ids.extend(edge.target for edge in self.out_edges.get(current, []))
            for nid in neighbor_ids:
                if nid in seen or nid not in self.nodes:
                    continue
                seen.add(nid)
                node = self.nodes[nid]
                if self.is_article_node(node):
                    return node
                q.append((nid, depth + 1))
        return None

    def nearest_clause(self, node_id: Optional[str], max_depth: int = 5) -> Optional[Node]:
        if not node_id or node_id not in self.nodes:
            return None
        if self.is_clause_node(self.nodes[node_id]):
            return self.nodes[node_id]
        seen = {node_id}
        q = deque([(node_id, 0)])
        while q:
            current, depth = q.popleft()
            if depth >= max_depth:
                continue
            # Ưu tiên đi ngược lên cha trước, vì điểm/hình phạt thường nằm dưới khoản.
            neighbor_ids = [e.source for e in self.in_edges.get(current, [])] + [e.target for e in self.out_edges.get(current, [])]
            for nid in neighbor_ids:
                if nid in seen or nid not in self.nodes:
                    continue
                seen.add(nid)
                node = self.nodes[nid]
                if self.is_clause_node(node):
                    return node
                q.append((nid, depth + 1))
        return None

    def nearest_point(self, node_id: Optional[str], max_depth: int = 4) -> Optional[Node]:
        if not node_id or node_id not in self.nodes:
            return None
        if self.is_point_node(self.nodes[node_id]):
            return self.nodes[node_id]
        seen = {node_id}
        q = deque([(node_id, 0)])
        while q:
            current, depth = q.popleft()
            if depth >= max_depth:
                continue
            neighbor_ids = [e.source for e in self.in_edges.get(current, [])] + [e.target for e in self.out_edges.get(current, [])]
            for nid in neighbor_ids:
                if nid in seen or nid not in self.nodes:
                    continue
                seen.add(nid)
                node = self.nodes[nid]
                if self.is_point_node(node):
                    return node
                q.append((nid, depth + 1))
        return None

    def reference_of_node(self, node_id: Optional[str], max_depth: int = 5) -> Dict[str, object]:
        """Trả về reference pháp lý gần nhất ở mức Điểm/Khoản/Điều.

        Đây là phần quan trọng để score không dừng ở Khoản: mỗi candidate được gắn với
        article + clause + point gần nhất, sau đó rerank/evidence có thể aggregate tới Điểm.
        """
        node = self.nodes.get(node_id or "")
        article = self.nearest_article(node_id, max_depth=max_depth)
        clause = None
        point = None

        # Không để node Điều tự kéo xuống Khoản/Điểm lân cận; Điều phải là Điều.
        if node and self.is_article_node(node):
            article = node
        elif node and self.is_clause_node(node):
            clause = node
        elif node and self.is_point_node(node):
            point = node
            clause = self.nearest_clause(node_id, max_depth=max_depth)
        else:
            # Node con như hình phạt/hành vi/tình tiết: tìm Điểm trước, rồi Khoản gần nhất.
            point = self.nearest_point(node_id, max_depth=max_depth)
            clause = self.nearest_clause(node_id, max_depth=max_depth)

        art_num = self.article_number_of_node(article) if article else None
        clause_num = self.clause_number_of_node(clause) if clause else None
        point_chr = self.point_letter_of_node(point) if point else None

        display = legal_reference_display(art_num, clause_num, point_chr)
        key_parts = [f"article:{art_num or ''}"]
        if clause_num:
            key_parts.append(f"clause:{clause_num}")
        if point_chr:
            key_parts.append(f"point:{point_chr}")
        return {
            "article": art_num,
            "clause": clause_num,
            "point": point_chr,
            "article_id": article.id if article else None,
            "clause_id": clause.id if clause else None,
            "point_id": point.id if point else None,
            "target_node_id": (point.id if point else clause.id if clause else article.id if article else node_id),
            "display": display,
            "key": "|".join(key_parts),
            "specificity": (1 if art_num else 0) + (1 if clause_num else 0) + (1 if point_chr else 0),
        }

    def same_reference(self, node_id: Optional[str], reference: Dict[str, object], level: str = "point") -> bool:
        if not node_id or not reference:
            return False
        ref = self.reference_of_node(node_id)
        if reference.get("article") and ref.get("article") != reference.get("article"):
            return False
        if level in {"clause", "point"} and reference.get("clause") and ref.get("clause") != reference.get("clause"):
            return False
        if level == "point" and reference.get("point") and ref.get("point") != reference.get("point"):
            return False
        return True

    def shortest_distance(self, start: str, target: str, max_depth: int = 5) -> Optional[int]:
        if start == target:
            return 0
        if start not in self.nodes or target not in self.nodes:
            return None
        q = deque([(start, 0)])
        seen = {start}
        while q:
            current, depth = q.popleft()
            if depth >= max_depth:
                continue
            neighbors = [e.target for e in self.out_edges.get(current, [])] + [e.source for e in self.in_edges.get(current, [])]
            for nid in neighbors:
                if nid == target:
                    return depth + 1
                if nid in seen or nid not in self.nodes:
                    continue
                seen.add(nid)
                q.append((nid, depth + 1))
        return None

    def paths_from_node(self, node_id: Optional[str], depth: int = 2, max_paths: int = 8) -> List[str]:
        if not node_id or node_id not in self.nodes:
            return []
        paths: List[str] = []
        start = self.nodes[node_id]
        q = deque([(node_id, short_text(start.name, 120), 0)])
        seen = {(node_id, 0)}
        while q and len(paths) < max_paths:
            current, path_text, d = q.popleft()
            if d >= depth:
                continue
            for edge in self.out_edges.get(current, []):
                if edge.target not in self.nodes:
                    continue
                nxt = self.nodes[edge.target]
                next_text = f"{path_text} --{edge.relation or 'liên quan'}--> {short_text(nxt.name, 120)}"
                paths.append(next_text)
                state = (edge.target, d + 1)
                if state not in seen:
                    seen.add(state)
                    q.append((edge.target, next_text, d + 1))
        return paths

    def find_penalties_near(self, node_id: Optional[str], depth: int = 4) -> List[Node]:
        if not node_id or node_id not in self.nodes:
            return []
        expanded = self.expand([node_id], depth=depth, max_nodes=180)
        return [node for node in expanded if self.is_penalty_node(node)]

    def article_family(self, article_id: Optional[str], depth: int = 3) -> List[Node]:
        if not article_id or article_id not in self.nodes:
            return []
        return self.expand([article_id], depth=depth, max_nodes=160)
