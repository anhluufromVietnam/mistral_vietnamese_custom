# mistral_vietnamese_custom with llama_cpp GPU Benchmark & Chat Interface

This project provides a complete setup for benchmarking, plotting, and chatting with `llama_cpp` using GPU acceleration (CUDA-only, no cuDNN required). It includes:

- `benchmark.py`: Benchmark llama models with different `ctx`, `gpu_layers`, and `batch` settings.
- `plotmat.py`: Visualize the benchmark results.
- `AI.py`: Interact with the model through a terminal chat interface.
- `testgpu.py`: Verify GPU status and VRAM usage.

---

## 🚀 Installation

### ✅ Requirements

- Windows 10/11 with an **NVIDIA GPU**
- Python 3.9+
- CUDA Toolkit 12.x (cuDNN **not required**)
- Visual Studio C++ Build Tools

### 🔧 Setup Instructions

1. **Install Python**  
   Download from: https://www.python.org/downloads/

2. **Install Visual Studio Build Tools**  
   - Get it here: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select: **Desktop Development with C++**

3. **Install CUDA Toolkit**  
   Download from: https://developer.nvidia.com/cuda-downloads

4. **Ensure Latest NVIDIA GPU Drivers**  
   Update from: https://www.nvidia.com/Download/index.aspx

5. **Clone this repository**  
   ```bash
   git clone [https://github.com/anhluufromVietnam/mistral_vietnamese_custom.git]
   cd mistral_vietnamese_custom
Install dependencies

# Llama Benchmark GPU

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install `llama_cpp` (CUDA-enabled):
   If you have the `.whl` file (`llama_cpp_python-0.3.4-cp39-cp39-win_amd64.whl`), install it directly:
   ```bash
   pip install llama_cpp_python-0.3.4-cp39-cp39-win_amd64.whl
   ```

## 📁 Scripts & Usage

### 🧪 `benchmark.py`
Run benchmarks for multiple configuration combinations.
```bash
python benchmark.py
```
This will:
- Run inference with various context sizes (`ctx`), GPU layers (`gpu_layers`), and batch sizes.
- Save results to `benchmark_output.csv`.
- Automatically release GPU memory between runs.

### 📈 `plotmat.py`
Visualize benchmark results as graphs.
```bash
python plotmat.py
```
You’ll get:
- One subplot per context size.
- Lines showing how response time varies by batch and GPU layers.
- Colorful, connected lines for easy comparison.

### 💬 `AI.py`
Simple chat interface with the model.
```bash
python AI.py
```
You can enter your questions and get AI responses powered by the local `llama_cpp` backend.

### 🧠 `testgpu.py`
Check whether your GPU is active and which processes are using VRAM.
```bash
python testgpu.py
```

## 📊 Benchmark Results Summary
Based on the `benchmark_output.csv` data, here are average response times:
### 🧠 Benchmark Results Summary for 16-bit floating point precision

| Context Size | GPU Layers | Batch Size  | Avg Response Time (sec) |
|--------------|------------|-------------|--------------------------|
| 2048         | 20         | 256–1024    | ~23.1                    |
| 2048         | 24         | 256–1024    | ~15.5                    |
| 2048         | 28         | 256–1024    | ~12.2                    |
| 2048         | 32         | 256–1024    | ~15.2                    |
| 4096–16384   | 28         | 512–1024    | ~12.2                    |
| 32768        | >28        | Any         | ❌ Failed (VRAM exceeded) |
### 🧠 Benchmark Results Summary for 4-bit Quantization
| Context Size | GPU Layers | Batch Size  | Avg Response Time (sec) |
|--------------|------------|-------------|--------------------------|
| 2048         | 20         | 256–1024    | ~7.3                     |
| 2048         | 24         | 256–1024    | ~5.7                     |
| 2048         | 28         | 256–1024    | ~3.6                     |
| 2048         | 32         | 256–1024    | ~2.0                     |
| 4096         | 28         | 512–1024    | ~3.6                     |
| 8192         | 32         | 256–1024    | ~2.0                     |
| 16384        | 32         | 256–1024    | ~2.0                     |
| 32768        | 32         | 256–1024    | ~2.0                     |

## ✅ Best Performance
### for 16-bit floating point precision
- **GPU Layers**: 28
- **Batch**: 1024
- **Ctx**: 2048 to 16384
- Delivers ~12.2 sec response time on average.

## ⚠️ Limitations
- At `ctx=32768`, models with 28+ GPU layers fail due to memory exhaustion (~75MB allocation failure).
- Increase in GPU layers beyond 28 does not always yield better performance and may reduce VRAM headroom.

## ✅ Best Performance
### for 4-bit quantization (q4_1)
GPU Layers: 32
Batch: 256 to 1024
Ctx: 2048 to 32768
Delivers ~2.0 sec response time on average.

## ⚠️ Limitations
At ctx=32768, models with more than 28 GPU layers fail due to memory exhaustion.
Response times may vary with lower GPU layers, indicating potential trade-offs in performance.

### Response Time Summary Table


| Context Size | GPU Layers | Batch Size | Avg Response Time (sec) |
|--------------|------------|------------|--------------------------|
| 16384        | 32         | 512        | 2.09                     |
| 16384        | 32         | 1024       | 2.02                     |
| 16384        | 36         | 512        | 1.67                     |
| 16384        | 36         | 1024       | 1.66                     |
| 16384        | 40         | 512        | 1.66                     |
| 16384        | 40         | 1024       | 1.66                     |
| 16384        | 44         | 512        | 1.66                     |
| 16384        | 44         | 1024       | 1.67                     |
| 16384        | 48         | 512        | 1.67                     |
| 16384        | 48         | 1024       | 1.67                     |
| 32768        | 32         | 512        | 2.05                     |
| 32768        | 32         | 1024       | 2.05                     |
| 32768        | 36         | 512        | 1.69                     |
| 32768        | 36         | 1024       | 1.69                     |
| 32768        | 40         | 512        | 1.69                     |
| 32768        | 40         | 1024       | 1.70                     |
| 32768        | 44         | 512        | 1.69                     |
| 32768        | 44         | 1024       | 1.69                     |
| 32768        | 48         | 512        | 1.70                     |
| 32768        | 48         | 1024       | 1.69                     |


### Conclusion

## ✅ Optimal Performance
- **GPU Layers**: 44 and 48
- **Batch**: 1024
- **Ctx**: 16384 to 32768
- Delivers response times around **1.66 to 1.67 seconds**.

## 📊 Key Insights
- **Impact of Batch Size**: Increasing batch size to 1024 improves response times, particularly at higher GPU layers.
- **Scalability**: As context size increases, response times stabilize, indicating efficient scalability for larger inputs.
- **Layer Efficiency**: Performance gains plateau beyond 36 GPU layers, suggesting diminishing returns with higher layers.

In summary, using 44 or 48 GPU layers with a batch size of 1024 provides the best performance balance for larger context sizes.


# 🧠 Vistral 7B (4Q\_1) Inference Benchmark

This document presents benchmark results evaluating **response time** across different context sizes, batch sizes, and GPU layer offloading levels using **Vistral 7B quantized to 4-bit (4Q\_1)**.

---

## 📊 Conclusion from the Graph

### Response Time vs Context Size:

* Response time is relatively **stable** across context sizes (16,384 vs 32,768).
* Slightly **higher response time** occurs at lower context sizes for **smaller batch sizes** and **fewer GPU layers**.

### Effect of Batch Size:

* **Batch 2048** generally results in **slightly lower response times** than batch 1024.

### Effect of GPU Layers:

* Increasing GPU layers (e.g., 128 → 1024) **slightly reduces response time**, especially for **larger batches**.

---

## 📋 Benchmark Summary Table

| Context Size | GPU Layers | Batch Size | Avg Response Time (sec)  |
| ------------ | ---------- | ---------- | ------------------------ |
| 16384        | 64         | 1024–2048  | \~1.71                   |
| 16384        | 128        | 1024–2048  | \~1.67                   |
| 16384        | 256        | 1024–2048  | \~1.67                   |
| 16384        | 1024       | 2048       | \~1.66                   |
| 32768        | 256        | 1024–2048  | \~1.69                   |
| 32768        | 1024       | 1024–2048  | \~1.70                   |
| 32768        | >1024      | Any        | ❌ Failed (VRAM exceeded) |

---


## 📌 File Structure
```bash
llama-benchmark-gpu/
├── benchmark.py                      # Run benchmark tests
├── plotmat.py                        # Plot results from benchmark_output.csv
├── AI.py                             # Interactive chat with LLM
├── testgpu.py                        # GPU usage check
├── benchmark_output.csv
├── requirements.txt
├── ggml-vistral-7B-chat-f16.gguf     # Artifacts model file
└── README.md
```

## 🧠 Model Info & Credits

We use the **[Vistral-7B-Chat](https://huggingface.co/Vistral-7B-Chat-GGUF)** model, a cutting-edge Vietnamese large language model based on the **Mistral architecture**.

### 📌 Model Overview:
- **Architecture**: Based on [Mistral-7B](https://arxiv.org/abs/2310.06825), adapted for Vietnamese.
- **Training**: Fine-tuned with high-quality Vietnamese instructions and conversation data.
- **Use Case**: Chatbot applications, question answering, summarization, and general Vietnamese NLP tasks.
- **Format**: GGUF format for efficient inference with `llama.cpp` and `llama-cpp-python`.

### 🧾 Citation

If you use this model in your work, please cite:

```bash
@article{chien2023vistral,
  author = {Chien Van Nguyen, Thuat Nguyen, Quan Nguyen, Huy Huu Nguyen, Björn Plüster, Nam Pham, Huu Nguyen, Patrick Schramowski, Thien Huu Nguyen},
  title = {Vistral-7B-Chat - Towards a State-of-the-Art Large Language Model for Vietnamese},
  year = 2023,
}
```
---

### 🏁 Run the server

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

### 📊 Performance Table for chat USING 

|  # | Prompt Tokens | Gen Tokens | Total Tokens | Total Time (ms) | Time per Token (ms) |
| -: | ------------- | ---------- | ------------ | --------------- | ------------------- |
|  1 | 53            | 91         | 144          | 1818.66         | 12.63               |
|  2 | 9             | 37         | 46           | 916.86          | 19.93               |
|  3 | 11            | 68         | 79           | 1312.16         | 16.61               |
|  4 | 26            | 511        | 537          | 9187.08         | 17.10               |
|  5 | 28            | 223        | 251          | 4135.72         | 16.47               |
|  6 | 481           | 511        | 992          | 9539.92         | 9.62                |
|  7 | 435           | 511        | 946          | 9392.18         | 9.93                |
|  8 | 15            | 22         | 37           | 736.83          | 19.91               |
|  9 | 13            | 34         | 47           | 845.04          | 17.98               |
| 10 | 16            | 49         | 65           | 1183.84         | 18.21               |
| 11 | 20            | 55         | 75           | 1198.02         | 15.97               |
| 12 | 16            | 51         | 67           | 1087.73         | 16.23               |

---

### 📝 GitHub-Style Conclusion

**Model Performance Summary**

* ✅ **Model Load Time:** Consistently around **324 ms**.
* ✅ **Total Inference Time per Token:** Ranges between **9.6–19.9 ms/token** depending on prompt/gen length.
* ✅ **Peak Efficiency:** Achieved on larger token batches (e.g. 992 tokens @ \~9.6 ms/token).
* ⚠️ **Zero ms for eval/prompt times:** Likely indicates that internal timing is not recorded properly in `llama-cpp-python` under current backend or threading config.
* ❗ **`GGML_ASSERT(...) failed` crash earlier:** A sign of tensor graph inconsistency, potentially caused by:

  * Corrupted KV cache reuse
  * Incorrect context reuse between threads
  * Mismatch in graph setup

**Recommendations**

* ✅ Acceptable latency for local inference
* 🧪 Test with `--n-gpu-layers 0` and `--threads` to isolate performance bottlenecks
* ⚙️ Ensure KV cache reuse is handled properly (especially with async calls)
* 🛠️ Consider upgrading to latest `llama-cpp-python` with patched `ggml` backend
* 🧹 Clear context between conversations to avoid graph assertion errors

---


📞 Contact
<<<<<<< HEAD
For questions or contributions, please open an issue or PR.
=======
For questions or contributions, please open an issue or PR.
>>>>>>> c83079374d6febec17fe1c6996d24199558036ad
