from huggingface_hub import InferenceClient


client = InferenceClient()


response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is exploratory data analysis?"
        }
    ],
    max_tokens=150
)


print(response.choices[0].message.content)