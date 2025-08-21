import chromadb
from rich.console import Console
console = Console()
from typing import Dict, List

class LongTermMemory:
    def __init__(self, collection_name: str, client: chromadb.PersistentClient):
        self.client = client
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def test_connection(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            console.print(f"Erreur de connexion ChromaDB: {str(e)}")
            return False

    def upsert_memory(self, content: str, metadata: dict, id: str):
        self.collection.upsert(documents=[content], metadatas=[metadata], ids=[id])

    def add_memory(self, content: str, metadata: dict):
        # This is a simplified add_memory, in a real scenario you might generate an ID
        # or handle duplicates differently.
        self.collection.add(documents=[content], metadatas=[metadata], ids=[str(hash(content))])

    def query_memory(self, query_texts: List[str], n_results: int = 10) -> List[Dict]:
        results = self.collection.query(query_texts=query_texts, n_results=n_results)
        return results.get("documents", [])


