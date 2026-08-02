"""
Knowledge graph storage and query-time retrieval: an in-memory NetworkX
MultiDiGraph for fast traversal, backed by SQLite for durability. This
replaces the earlier Neo4j-based design -- no server, no Docker, same
conceptual role (store triples, traverse neighborhoods at query time).

Write-through pattern: every write (from ingestion) goes to SQLite AND
the in-memory graph in the same call, so the graph is always current and
SQLite is always the durable source of truth. On process startup, the
whole graph is loaded from SQLite once into memory; reads never touch
disk after that.

Two caveats worth knowing, not discovering later:
- Concurrency: writes are guarded by a lock, since FastAPI can process
  requests concurrently even in a single process.
- Single-process assumption: running multiple uvicorn workers would give
  each its own in-memory graph, silently diverging from SQLite and from
  each other. Fine for a learning/POC deployment; revisit before scaling
  workers.
"""
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate

from Config.llm_config import fast_llm
from Config.settings import settings
from Utils.Logger import get_logger

logger = get_logger(__name__)



_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    source_doc TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    relation TEXT NOT NULL,
    target TEXT NOT NULL,
    source_doc TEXT
);
"""


class NetworkXGraphStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        Path(settings.graph_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(settings.graph_db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self._graph = nx.MultiDiGraph()
        self._load_from_db()
        logger.info(
            "Graph store initialized: %d nodes, %d edges loaded from %s",
            self._graph.number_of_nodes(), self._graph.number_of_edges(), settings.graph_db_path,
        )

    def _load_from_db(self) -> None:
        cur = self._conn.cursor()
        for node_id, source_doc in cur.execute("SELECT id, source_doc FROM nodes"):
            self._graph.add_node(node_id, source_doc=source_doc)
        for source, relation, target in cur.execute(
            "SELECT source, relation, target FROM edges"
        ):
            self._graph.add_edge(source, target, key=relation, relation=relation)

    def add_triples(self, triples: List[Tuple[str, str, str]], source_doc: str) -> None:
        """triples: list of (source_entity, relation, target_entity)."""
        if not triples:
            logger.debug("add_triples called with no triples for '%s'", source_doc)
            return
        with self._lock:
            cur = self._conn.cursor()
            for source, relation, target in triples:
                cur.execute(
                    "INSERT OR IGNORE INTO nodes (id, source_doc) VALUES (?, ?)",
                    (source, source_doc),
                )
                cur.execute(
                    "INSERT OR IGNORE INTO nodes (id, source_doc) VALUES (?, ?)",
                    (target, source_doc),
                )
                cur.execute(
                    "INSERT INTO edges (source, relation, target, source_doc) VALUES (?, ?, ?, ?)",
                    (source, relation, target, source_doc),
                )
                self._graph.add_node(source, source_doc=source_doc)
                self._graph.add_node(target, source_doc=source_doc)
                self._graph.add_edge(source, target, key=relation, relation=relation)
            self._conn.commit()
            logger.info("Added %d triple(s) to graph store for '%s'", len(triples), source_doc)

    def find_entity(self, mention: str) -> Optional[str]:
        """
        Case-insensitive substring match against node ids -- the NetworkX
        equivalent of Cypher's `WHERE toLower(e.id) CONTAINS toLower($x)`.
        Returns the first match, or None. A lowercase name index would be
        the next optimization once the graph grows large; a linear scan
        is fine at POC scale.
        """
        mention_lower = mention.lower()
        for node_id in self._graph.nodes:
            if mention_lower in node_id.lower():
                return node_id
        return None

    def neighbor_facts(self, mention: str, depth: int = 1, limit: int = 20) -> List[str]:
        """
        The NetworkX equivalent of Cypher's `MATCH (e)-[r]-(neighbor)`:
        find the entity, then walk its neighborhood out to `depth` hops.
        """
        node_id = self.find_entity(mention)
        if node_id is None:
            return []

        subgraph = nx.ego_graph(self._graph, node_id, radius=depth, undirected=True)
        facts: List[str] = []
        for source, target, data in subgraph.edges(data=True):
            facts.append(f"{source} -[{data.get('relation', '?')}]-> {target}")
            if len(facts) >= limit:
                break
        return facts

    def all_facts(self, limit: int = 50) -> List[str]:
        """
        Returns a sample of every relationship in the graph, capped at
        `limit`. Used as a fallback when a graph-strategy question doesn't
        name any specific entity to anchor a neighborhood search on (e.g.
        "give me all the relationships") -- rather than returning nothing,
        surface what's actually in the graph.
        """
        facts: List[str] = []
        for source, target, data in self._graph.edges(data=True):
            facts.append(f"{source} -[{data.get('relation', '?')}]-> {target}")
            if len(facts) >= limit:
                break
        return facts


_graph_store_singleton: Optional[NetworkXGraphStore] = None


def get_graph_store() -> NetworkXGraphStore:
    global _graph_store_singleton
    if _graph_store_singleton is None:
        _graph_store_singleton = NetworkXGraphStore()
    return _graph_store_singleton


# --- Query-time retrieval (used by the LangGraph agent graph) ---

_ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the key named entities (people, organizations, concepts, "
            "products) mentioned in the question. Return them as a "
            "comma-separated list and nothing else -- no explanation. "
            "If there are no relevant named entities, return exactly the "
            "word NONE and nothing else.",
        ),
        ("human", "{question}"),
    ]
)

def _extract_entities(question: str) -> List[str]:
    chain = _ENTITY_PROMPT | fast_llm
    raw = chain.invoke({"question": question}).content.strip()
    if raw.upper() == "NONE":
        logger.info("No entities found in question: %r", question[:80])
        return []
    entities = [e.strip() for e in raw.split(",") if e.strip()]
    logger.info("Extracted entities from question: %s", entities)
    return entities


def graph_search(question: str, depth: int = 1) -> List[str]:
    """
    Local search: pull entities mentioned in the question, then gather
    each one's neighborhood facts from the graph store.
    """
    store = get_graph_store()
    entities = _extract_entities(question)

    logger.info(store.all_facts())

    facts: List[str] = []
    seen = set()
    for entity in entities:
        for fact in store.neighbor_facts(entity, depth=depth):
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)

    logger.info("Graph search for %r returned %d fact(s)", question[:80], len(facts))
    return facts
