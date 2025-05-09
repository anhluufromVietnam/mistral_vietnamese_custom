import time
import os
import csv
from llama_cpp import Llama

# Benchmark configurations
ctx_values = [2048, 4096, 8192, 16384, 32768]
gpu_layer_values = [20, 24, 28, 32]
batch_values = [256, 512, 1024]

# Model path
model_path = "ggml-vistral-7B-chat-f16.gguf"

# Sample prompt
sample_prompt = "Hãy mô tả ngắn gọn về cách hoạt động của trí tuệ nhân tạo trong đời sống hàng ngày."

# Output file
output_file = "llama_benchmark_results.csv"

# Ensure CSV header
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["ctx", "gpu_layers", "batch", "load_time", "response_time", "error"])

def kill_gpu_processes():
    # List processes using GPU and kill them
    os.system('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | PowerShell -Command "foreach ($pid in ($_)) {Stop-Process -Id $pid -Force}"')
    print("✅ GPU processes have been killed!")

# Benchmark execution
for n_ctx in ctx_values:
    for n_gpu_layers in gpu_layer_values:
        for n_batch in batch_values:
            print(f"\n=== Benchmark: ctx={n_ctx}, gpu_layers={n_gpu_layers}, n_batch={n_batch} ===")
            load_time = response_time = 0
            error_msg = ""

            try:
                # Load model
                start_load = time.time()
                llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=8,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                load_time = round(time.time() - start_load, 2)

                # Run inference
                prompt = f"<s>[INST] {sample_prompt} [/INST]"
                start = time.time()
                output = llm(prompt, max_tokens=100, stop=["</s>"])
                response_time = round(time.time() - start, 2)

                response = output["choices"][0]["text"].strip()
                print(f"✅ Model Load Time: {load_time} seconds")
                print(f"✅ Response Time: {response_time} seconds")
                print(f"🗣️ {response[:200]}...\n")

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error: {error_msg}")

            # Log result to CSV
            with open(output_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([n_ctx, n_gpu_layers, n_batch, load_time, response_time, error_msg])

            # Cleanup
            kill_gpu_processes()
            llm = None
            print("✅ VRAM has been freed!")
