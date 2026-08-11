# Synthetic recovery fixture: the missing colon is intentional.
class RepositoryIndexer:
    def __init__(self, embedder):
        self.embedder = embedder

    def index(self, documents):
        vectors = []
        for document in documents
            vectors.append(self.embedder.embed(document))
        return vectors

    def count(self, documents):
        return len(documents)
