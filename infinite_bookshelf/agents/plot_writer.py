"""
Agent to generate novel plot structure
"""

from ..inference import GenerationStatistics

def generate_plot_structure(
    prompt: str,
    characters: str,
    genre: str,
    narrative_style: str,
    additional_instructions: str,
    model: str,
    groq_provider
):
    """
    Generate a narrative arc and plot structure for the novel.
    Returns plot structure in JSON format and generation statistics.
    """
    USER_PROMPT = f"""
    Create a detailed plot structure for a novel with the following, and return the result in JSON format:
    
    <concept>{prompt}</concept>
    <genre>{genre}</genre>
    <narrative_style>{narrative_style}</narrative_style>
    <characters>{characters}</characters>
    <additional_instructions>{additional_instructions}</additional_instructions>
    
    Create a structure with exposition, rising action, complications, climax, and resolution.
    For each plot point, explain what happens and which characters are involved.
    """
    
    completion = groq_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'Structure your response as a JSON object with chapter titles and plot points. Follow the dramatic arc, ensuring character development and proper pacing. Format example: {"Chapter 1: Title": "Plot description", "Chapter 2: Title": {"Scene 1": "Scene description", "Scene 2": "Scene description"}}',
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],
        temperature=0.6,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics = GenerationStatistics(
        input_time=usage.prompt_time,
        output_time=usage.completion_time,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_time=usage.total_time,
        model_name=model,
    )

    return statistics, completion.choices[0].message.content 