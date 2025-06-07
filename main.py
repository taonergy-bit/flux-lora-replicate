from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import requests
import os
from uuid import uuid4

app = FastAPI()

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16
).to("cuda")


class Request(BaseModel):
    prompt: str
    width: int = 768
    height: int = 1024
    lora_url: str


@app.post("/generate")
def generate(req: Request):
    # Download LoRA
    lora_filename = f"/tmp/{uuid4().hex}.safetensors"
    r = requests.get(req.lora_url)
    with open(lora_filename, "wb") as f:
        f.write(r.content)

    # Load LoRA and run
    pipe.load_lora_weights(lora_filename)
    image = pipe(prompt=req.prompt, width=req.width, height=req.height).images[0]

    path = f"/tmp/{uuid4().hex}.png"
    image.save(path)

    return {"path": path}
