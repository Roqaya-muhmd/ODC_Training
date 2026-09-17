from rag import create_vector_store
from huggingface_hub import InferenceClient


print("Creating vector store...")

vector_store = create_vector_store()

print("Vector store ready! ✅")


question = "What is the difference between a distribution plot and a count plot?"


results = vector_store.similarity_search(
    question,
    k=2
    
    
)


context = "\n\n".join(
    result.page_content
    for result in results
)


prompt = f"""
You are a data analysis assistant.

Answer the user's question using the provided context.

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""


client = InferenceClient()

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=250
)


print("\n===== FINAL ANSWER =====")
print(response.choices[0].message.content)