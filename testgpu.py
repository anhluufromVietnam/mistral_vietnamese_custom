from llama_cpp import Llama

llm = Llama(model_path="ggml-vistral-7B-chat-f16.gguf", n_gpu_layers=30)
print("GPU layers:", llm.context_params["n_gpu_layers"])
