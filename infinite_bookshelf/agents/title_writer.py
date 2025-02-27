"""
Agent to generate book title
"""

from ..inference import GenerationStatistics


def generate_book_title(prompt: str, model: str, groq_provider):
    """
    Generate a book title using AI.
    """
    completion_params = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert book title creator. Generate a compelling, intriguing title that captures the essence of a novel concept.",
            },
            {
                "role": "user",
                "content": f"Create a captivating title for a novel with this concept: {prompt}\n\nReturn only the title, nothing else.",
            },
        ],
        "temperature": 0.8,
        "max_tokens": 50,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    
    if "deepseek" in model.lower():
        completion_params["reasoning_format"] = "hidden"
    
    completion = groq_provider.chat.completions.create(**completion_params)
    
    return completion.choices[0].message.content.strip().strip('"')
