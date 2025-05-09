# mistral_vietnamese_custom

# 🦙 llama_cpp GPU Benchmark & Chat Interface

This project provides a complete setup for benchmarking, plotting, and chatting with `llama_cpp` using GPU acceleration (CUDA-only, no cuDNN required). It includes:

- `benchmark.py`: Benchmark llama models with different `ctx`, `gpu_layers`, and `batch` settings.
- `plotmat.py`: Visualize the benchmark results.
- `AI.py`: Interact with the model through a chat interface.
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
   git clone https://github.com/your-username/llama-benchmark-gpu.git
   cd llama-benchmark-gpu
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Install llama_cpp (CUDA-enabled)
If you have the .whl file (llama_cpp_python-0.3.4-cp39-cp39-win_amd64.whl), install it directly:

bash
Copy
Edit
pip install llama_cpp_python-0.3.4-cp39-cp39-win_amd64.whl
📁 Scripts & Usage
🧪 benchmark.py
Run benchmarks for multiple configuration combinations.

bash
Copy
Edit
python benchmark.py
This will:

Run inference with various context sizes (ctx), GPU layers (gpu_layers), and batch sizes

Save results to benchmark_output.csv

Automatically release GPU memory between runs

📈 plotmat.py
Visualize benchmark results as graphs.

bash
Copy
Edit
python plotmat.py
You’ll get:

One subplot per context size

Lines showing how response time varies by batch and GPU layers

Colorful, connected lines for easy comparison

💬 AI.py
Simple chat interface with the model.

bash
Copy
Edit
python AI.py
You can enter your questions and get AI responses powered by the local llama_cpp backend.

🧠 testgpu.py
Check whether your GPU is active and which processes are using VRAM.

bash
Copy
Edit
python testgpu.py
📊 Benchmark Results Summary
Based on the benchmark_output.csv data, here are average response times:

Context Size	GPU Layers	Batch Size	Avg Response Time (sec)
2048	20	256–1024	~23.1
2048	24	256–1024	~15.5
2048	28	256–1024	~12.2
2048	32	256–1024	~15.2
4096–16384	28	512–1024	~12.2
32768	>28	Any	❌ Failed (VRAM exceeded)

✅ Best Performance
GPU Layers = 28

Batch = 1024

Ctx = 2048 to 16384

Delivers ~12.2 sec response time on average.

⚠️ Limitations
At ctx=32768, models with 28+ GPU layers fail due to memory exhaustion (~75MB allocation failure).

Increase in GPU layers beyond 28 does not always yield better performance and may reduce VRAM headroom.

📌 File Structure
bash
Copy
Edit
llama-benchmark-gpu/
├── benchmark.py        # Run benchmark tests
├── plotmat.py          # Plot results from benchmark_output.csv
├── AI.py               # Interactive chat with LLM
├── testgpu.py          # GPU usage check
├── benchmark_output.csv
├── requirements.txt
└── README.md
❤️ Credits
llama_cpp

CUDA Toolkit by NVIDIA

📞 Contact
For questions or contributions, please open an issue or PR.
