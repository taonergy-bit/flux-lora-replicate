from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import os
import requests
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
def generate_image(body: Request):
    filename = body.lora_url.split("/")[-1]
    local_path = f"/tmp/{filename}"

    if not os.path.exists(local_path):
        r = requests.get(body.lora_url)
        with open(local_path, "wb") as f:
            f.write(r.content)

    pipe.load_lora_weights(local_path)

    image = pipe(prompt=body.prompt, width=body.width, height=body.height).images[0]

    image_path = f"/tmp/{uuid4().hex}.png"
    image.save(image_path)

    return {"url": f"https://fal.run/file{image_path}"}
