from huggingface_hub import InferenceClient

from src.rag import create_vector_store, add_dataset_to_store
from src.data_analysis import load_data, summarize_dataset

LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

client = InferenceClient()

conversation_history = []
conversation_summary = ""

SUMMARY_BATCH = 3   # how many old turns to fold into the summary at once
KEEP_RECENT = 2     # how many most-recent turns to always keep in full


def upload_dataset(vector_store, file_path):
    """Reads a CSV, summarizes it, and adds it to the vector store."""
    df = load_data(file_path)
    dataset_name = file_path.split("/")[-1]

    summary_text = summarize_dataset(df, dataset_name)
    add_dataset_to_store(vector_store, summary_text, dataset_name)

    print(f"Dataset '{dataset_name}' added to the knowledge base. ✅")
    return df


def retrieve_context(vector_store, question, k=6):
    """Same retrieval path for general questions AND dataset questions."""
    results = vector_store.similarity_search(question, k=k)
    return "\n\n".join(result.page_content for result in results)


def summarize_history():
    """
    Once there are enough old turns, folds them into conversation_summary
    via the LLM and trims conversation_history down to the recent turns.
    Same idea as the Summary notebook, just calling InferenceClient instead
    of a local model.
    """
    global conversation_summary
    global conversation_history

    if len(conversation_history) < (SUMMARY_BATCH + KEEP_RECENT):
        return  # not enough turns yet — nothing to summarize

    turns_to_summarize = conversation_history[:-KEEP_RECENT]
    remaining_turns = conversation_history[-KEEP_RECENT:]

    turns_text = ""
    for turn in turns_to_summarize:
        turns_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    prompt = f"""
Summarize the conversation below into a short paragraph that preserves
any facts the user shared (their name, their dataset, preferences, etc.).
Keep it concise.

Previous summary:
{conversation_summary if conversation_summary else "(none yet)"}

New turns to fold in:
{turns_text}

Updated summary:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    conversation_summary = response.choices[0].message.content.strip()
    conversation_history = remaining_turns

    print(f"[memory summarized — {len(turns_to_summarize)} turns folded in]")


def build_prompt(context, summary, history, question):
    """Combines retrieved context + summarized memory + recent turns + the new question."""

    history_text = ""
    for turn in history:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    summary_block = f"Summary of earlier conversation:\n{summary}\n\n" if summary else ""

    return f"""
You are a data analysis assistant.

Answer the user's question using the context below when it is relevant.
If the context does not contain the answer, say so and answer from
general knowledge instead.

Context:
{context}

{summary_block}Recent conversation:
{history_text}

Question:
{question}

Give a clear and simple answer.
"""


def chat(vector_store, question):
    context = retrieve_context(vector_store, question)
    prompt = build_prompt(context, conversation_summary, conversation_history, question)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )

    answer = response.choices[0].message.content

    conversation_history.append({"user": question, "assistant": answer})
    summarize_history()  # trims + folds in old turns once the threshold is hit

    return answer


def show_chat(vector_store, question):
    answer = chat(vector_store, question)
    print(f"\nUser     : {question}")
    print(f"Assistant: {answer}")


def print_memory_status():
    print("\n" + "=" * 60)
    print("MEMORY STATUS")
    print("=" * 60)
    print(f"Summary exists   : {'Yes' if conversation_summary else 'No'}")
    print(f"Recent turns kept: {len(conversation_history)}")
    if conversation_summary:
        print(f"\nSummary: {conversation_summary}")


if __name__ == "__main__":
    print("Creating vector store...")
    vector_store = create_vector_store()
    print("Vector store ready! ✅\n")

    # Example: uncomment to add a dataset before chatting
    upload_dataset(vector_store, "data/sample.csv")

    print("Chat started! Type 'quit' to stop, 'memory' to see memory status.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Chat ended.")
            break

        if user_input.lower() == "memory":
            print_memory_status()
            continue

        if not user_input:
            continue

        show_chat(vector_store, user_input)