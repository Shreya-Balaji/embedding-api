from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# Allow cross-origin requests (for Render to call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

class BatchTextsRequest(BaseModel):
    texts: List[str]

@app.post("/embed-batch")
async def embed_batch(req: BatchTextsRequest):
    # show_progress_bar=True will display progress in console logs where this app runs
    embeddings = model.encode(req.texts, show_progress_bar=True)
    return {"embeddings": embeddings.tolist()}
