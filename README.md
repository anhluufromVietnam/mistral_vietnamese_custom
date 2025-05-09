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

```bibtex
@article{chien2023vistral,
  author = {Chien Van Nguyen, Thuat Nguyen, Quan Nguyen, Huy Huu Nguyen, Björn Plüster, Nam Pham, Huu Nguyen, Patrick Schramowski, Thien Huu Nguyen},
  title = {Vistral-7B-Chat - Towards a State-of-the-Art Large Language Model for Vietnamese},
  year = 2023,
}


📞 Contact
For questions or contributions, please open an issue or PR.
