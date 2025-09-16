import ollama
import time

# Global client for connection reuse
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = ollama.Client()
    return _client

def llm(
    prompt: str, 
    model: str = 'gemma3:270m',
    system_prompt: str = "Answer the questions in one single sentence"
) -> str:
    # Handle both string input and message list input
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    # Use the global client for connection reuse
    client = _get_client()
    response = client.chat(model=model, messages=messages)
    response = response['message']['content'].strip().replace("\n", "")
    return response