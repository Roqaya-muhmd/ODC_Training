print("TEST STARTED")

from rag import create_vector_store

print("Creating vector store...")

vector_store = create_vector_store()

print("Vector store created successfully! ✅")

results = vector_store.similarity_search(
    "What is exploratory data analysis?",
    k=2
)

for result in results:
    print("\n--- Result ---")
    print(result.page_content)

print("TEST FINISHED")