import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Script started successfully.")

class SemanticSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_embeddings = None

    def add_documents(self, documents):
        """Add documents to the search index"""
        self.documents.extend(documents)
        new_embeddings = self.model.encode(documents)
        if self.document_embeddings is None:
            self.document_embeddings = new_embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])

    def search(self, query, top_k=3):
        """Search for most similar documents to the query"""
        if not self.documents:
            return []
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        return results

# Example usage
search_engine = SemanticSearchEngine()
# Add some documents
with open("document.txt", "r") as f:
    documents = [line.strip() for line in f if line.strip()]
print(f"{len(documents)} documents loaded")

search_engine.add_documents(documents)

def get_bot_response(query):
    if not query:
        return "Please enter a query."
    results = search_engine.search(query, top_k=1)
    if results:
        result = results[0]
        return (
            f"**Best Match (Index {result['index']}):**\n\n"
            f"{result['document']}\n\n"
            f"**Similarity Score:** {result['similarity']:.3f}"
        )
    else:
        return "No documents found."

