"""
Agent to generate novel section content with narrative depth
"""

from ..inference import GenerationStatistics

def generate_novel_section(
    title: str,
    section_description: str,
    plot_context: str,
    characters: str,
    genre: str,
    tone: str,
    narrative_style: str,
    previous_sections_summary: str,
    additional_instructions: str,
    model: str,
    groq_provider
):
    """
    Generate immersive, narratively consistent novel content.
    Maintains character consistency and follows given tone/style.
    """
    SYSTEM_PROMPT = """
    You are an expert fiction writer. Write compelling, emotionally resonant narrative content that:
    
    1. Shows rather than tells when describing characters and settings
    2. Uses natural-sounding dialogue appropriate to each character
    3. Balances description, dialogue, and action
    4. Creates appropriate pacing for the scene's emotional tone
    5. Maintains consistent character voices and behaviors
    6. Advances both plot and character development
    
    Only output the narrative content, without additional explanations.
    """
    
    USER_PROMPT = f"""
    Write an engaging narrative section with the following parameters:
    
    <section_title>{title}</section_title>
    <section_description>{section_description}</section_description>
    
    <plot_context>{plot_context}</plot_context>
    <characters>{characters}</characters>
    <genre>{genre}</genre>
    <tone>{tone}</tone>
    <narrative_style>{narrative_style}</narrative_style>
    
    <previous_sections_summary>{previous_sections_summary}</previous_sections_summary>
    <additional_instructions>{additional_instructions}</additional_instructions>
    
    Create immersive content that advances the story while developing characters. 
    Balance dialogue, action, and description.
    Maintain consistent characterization with previously established traits.
    """
    
    stream = groq_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],
        temperature=0.8,  # Higher for creative fiction
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