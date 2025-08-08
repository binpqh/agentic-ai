from fastapi import FastAPI
from pydantic import BaseModel
from . import agent

app = FastAPI()

class TextInput(BaseModel):
    text: str
    action: str  # "translate" hoặc "explain"

@app.post("/agent")
def process(input: TextInput):
    if input.action == "translate":
        return {"result": agent.translate(input.text)}
    elif input.action == "explain":
        return {"result": agent.explain(input.text)}
    else:
        return {"error": "Invalid action"}
