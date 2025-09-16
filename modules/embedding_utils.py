import ollama

# Global client for connection reuse
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = ollama.Client()
    return _client

def embedding(prompt: str) -> list:
    client = _get_client()
    result = client.embeddings(model='nomic-embed-text:v1.5', prompt=prompt)
    return result['embedding']