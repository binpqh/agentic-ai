from transformers import pipeline

class AIService:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        self.pipe = pipeline("text-generation", model=model_name, device_map="auto")

    def ask(self, prompt):
        result = self.pipe(prompt, max_new_tokens=256, do_sample=True)
        return result[0]["generated_text"]
