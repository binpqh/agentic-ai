from .llm_client import call_llm

def translate(text: str, to_lang: str = "en") -> str:
    prompt = f"Dịch sang {to_lang.upper()}: {text}"
    return call_llm(prompt)

def explain(text: str, lang: str = "vi") -> str:
    prompt = f"""
Bạn là giáo viên tiếng Anh. Hãy phân tích câu sau, sửa lỗi (nếu có), dịch nghĩa sang tiếng Việt, và giải thích từ vựng.
Câu: "{text}"
"""
    return call_llm(prompt)
