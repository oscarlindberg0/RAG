from ollama import chat

response = chat(
    model='granite4.2',
    messages=[{'role': 'user', 'content': 'Hello'}]
)

print(response.message.content)