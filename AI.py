import time
import os
from llama_cpp import Llama

def guess_difficulty(text):
    length = len(text)
    technical_keywords = ["AI", "machine learning", "học máy", "mạng nơ-ron", "deep learning", "thuật toán", "xác suất", "tính toán", "LLM"]
    sub_questions = text.count("?")

    score = 0
    if length > 80:
        score += 1
    if any(kw.lower() in text.lower() for kw in technical_keywords):
        score += 1
    if sub_questions > 1:
        score += 1

    if score == 0:
        return "Easy"
    elif score == 1:
        return "Medium"
    else:
        return "Hard"

# Optionally set this to show detailed GPU logs
os.environ["LLAMA_CPP_LOG_LEVEL"] = "info"

# Try to load model with GPU support
n_gpu_layers = 20
try:
    llm = Llama(
        model_path="ggml-vistral-7B-chat-f16.gguf",
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=20,
        n_gpu_layers=n_gpu_layers
    )
    print(f"✅ Model loaded with GPU acceleration (n_gpu_layers={n_gpu_layers}).")
except Exception as e:
    print("⚠️ Failed to load model with GPU. Falling back to CPU.")
    print("Reason:", e)
    llm = Llama(
        model_path="ggml-vistral-7B-chat-f16.gguf",
        n_ctx=2048,
        n_threads=8
    )
    print("✅ Model loaded on CPU.")

# System prompt
system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn."

print("\n=== Vistral Chat Q&A (Vietnamese) ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("🧑 Bạn: ")
    if user_input.lower() == "exit":
        break

    difficulty = guess_difficulty(user_input)

    # Format input using Vistral template
    prompt = f"""[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"""
""

    # Time the response
    start_time = time.time()
    output = llm(prompt, max_tokens=512, stop=["</s>"])
    end_time = time.time()

    response = output["choices"][0]["text"]
    duration = round(end_time - start_time, 2)

    # Display result
    print(f"\n🤖 Trợ lí ({difficulty}): {response.strip()}")
    print(f"⏱️ Thời gian phản hồi: {duration} giây\n")
