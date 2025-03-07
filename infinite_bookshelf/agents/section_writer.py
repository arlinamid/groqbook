"""
Agent to generate novel section content
"""

from ..inference import GenerationStatistics


def generate_section(
    prompt: str, 
    plot_context: str, 
    characters: str, 
    tone: str,
    additional_instructions: str, 
    model: str, 
    groq_provider
):
    stream = groq_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert fiction writer. Generate compelling narrative content for the section provided. Follow the tone, character behaviors, and plot context. Focus on engaging dialogue, vivid descriptions, and natural character development.",
            },
            {
                "role": "user",
                "content": f"""Generate engaging narrative content for the following section:

<section_title>{prompt}</section_title>

<plot_context>{plot_context}</plot_context>
                
<characters>{characters}</characters>

<tone>{tone}</tone>

<additional_instructions>{additional_instructions}</additional_instructions>

Write immersive, emotionally resonant content that advances the plot while developing characters. Balance dialogue, action, and description.
""",
            },
        ],
        temperature=0.7,  # Higher for creative fiction
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            statistics_to_return = GenerationStatistics(
                input_time=usage.prompt_time,
                output_time=usage.completion_time,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_time=usage.total_time,
                model_name=model,
            )
            yield statistics_to_return
