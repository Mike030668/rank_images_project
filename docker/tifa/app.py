from fastapi import FastAPI, UploadFile
from PIL import Image
import io

# ✔ правильный импорт
from tifascore import TifaEvaluator

te = TifaEvaluator(model="blip2_base")   # cpu; для gpu → .to("cuda")

api = FastAPI()

@api.post("/tifa")
async def tifa_score(prompt: str, file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    score = te.evaluate(prompt, img)
    return {"score": score}
