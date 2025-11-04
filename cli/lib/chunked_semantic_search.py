import re

from .semantic_search import SemanticSearch
from .file_utils import save_cache, load_cache

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    @staticmethod
    def __chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
        split_text = re.split(r"(?<=[.!?])\s+", text)
        step = chunk_size - overlap
        return [' '.join(split_text[i:i + chunk_size])
            for i in range(0, len(split_text), step)]

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        all_chunks = []
        metadata = []

        for i, document in enumerate(self.documents):
            if not document['description']:
                continue
            chunks = self.__chunk(document['description'], 4, 1)
            all_chunks.extend(all_chunks)
            num_chunks = len(chunks)
            metadata.extend(
                    [{'movie_idx': i, 'chunk_idx': j, 'total_chunks': num_chunks}
                    for j in range(num_chunks)])
            
        self.embeddings = self.model.encode(
            all_chunks,
            convert_to_numpy=True,
            how_progress_bar=True
        )
        save_cache(self.embeddings, "chunk_embeddings.npy")
        