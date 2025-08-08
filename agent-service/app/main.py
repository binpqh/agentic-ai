from fastapi import FastAPI
from pydantic import BaseModel
import agent

app = FastAPI()

class TextInput(BaseModel):
    text: str
    action: str

@app.post("/agent")
def process(prompt: TextInput):
    if prompt.action == "translate":
        return {"result": agent.translate(prompt.text)}
    elif prompt.action == "explain":
        return {"result": agent.explain(prompt.text)}
    else:
        return {"error": "Invalid action"}
