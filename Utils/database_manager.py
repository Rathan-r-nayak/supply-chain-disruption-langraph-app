import sqlite3
import json

class SqliteKeyValueStore:
    """A lightweight persistent LTM store using a local SQLite file."""
    
    def __init__(self, db_path="banking_ltm.db"):
        # Connects to (or creates) the local SQLite file
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    namespace TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (namespace, key)
                )
            """)

    def put(self, namespace: tuple, key: str, value: dict):
        """Saves a new fact to the database."""
        ns_str = ".".join(namespace)
        val_str = json.dumps(value)
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO memories (namespace, key, value)
                VALUES (?, ?, ?)
            """, (ns_str, key, val_str))

    def search(self, namespace: tuple):
        """Retrieves facts for a specific user."""
        ns_str = ".".join(namespace)
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM memories WHERE namespace = ?", (ns_str,))
        rows = cursor.fetchall()
        
        # Format to match LangGraph Store item structure so it integrates seamlessly
        class Item:
            def __init__(self, val):
                self.value = val
                
        return [Item(json.loads(row[1])) for row in rows]

# -------------------------------------------------------------------------
# INITIALIZATION
# This is the exact variable that app.py is trying to import!
# -------------------------------------------------------------------------
ltm_store = SqliteKeyValueStore("banking_ltm.db")