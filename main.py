import os
import threading
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama

# Disable verbose logs
os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model once
llm = Llama(
    model_path="ggml-vistral-7B-chat-q4_1.gguf",  # Adjust your path here
    n_ctx=16384,
    n_threads=64,
    n_gpu_layers=128,
    n_batch=512
)

# Thread safety
llm_lock = threading.Lock()

# System prompt (Vietnamese assistant)
system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn."

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, user_input: str = Form(...)):
    full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"

    with llm_lock:
        output = llm(full_prompt, max_tokens=512, stop=["</s>"])
    
    reply = output["choices"][0]["text"].strip()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_input,
        "response": reply
    })
