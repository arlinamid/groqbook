"""
Agent to generate novel structure with proper dramatic arc
"""

from ..inference import GenerationStatistics

def generate_novel_structure(
    prompt: str,
    characters: str,
    genre: str,
    narrative_style: str,
    themes: str,
    has_twist: bool,
    complexity_level: str,
    additional_instructions: str,
    model: str,
    groq_provider
):
    """
    Generate a structured novel outline with proper dramaturgy.
    Returns novel structure in JSON format with chapters and key scenes.
    """
    # Create a prompt that incorporates dramaturgical principles
    twist_instruction = "Include a surprising plot twist" if has_twist else ""
    
    SYSTEM_PROMPT = """
    You are an expert novel structure designer. Create a compelling novel structure in JSON format following classical dramaturgy:
    
    1. EXPOSITION: Introduce characters, setting, and initial situation (10-15% of the novel)
    2. INCITING INCIDENT: The event that sets the story in motion
    3. RISING ACTION: Escalating conflicts and complications (50-60% of the novel)
    4. MIDPOINT: A major revelation or shift in perspective
    5. COMPLICATIONS: Stakes rise, challenges intensify
    6. CLIMAX: The highest point of tension where the main conflict comes to a head (75-80% point)
    7. RESOLUTION: Aftermath and tying up of loose ends (final 10-15%)
    
    Format your response as a nested JSON structure with chapters and scenes, including character involvement and emotional beats.
    
    Format example: 
    {
      "Chapter 1: Title": "Description focusing on exposition and character introduction",
      "Chapter 2: Title": {
        "Scene 1: Setting": "Scene description with character dynamics",
        "Scene 2: Conflict": "Scene description with emerging tensions"
      }
    }
    """
    
    USER_PROMPT = f"""
    Create a detailed novel structure with the following parameters, returning a structured JSON format:
    
    <concept>{prompt}</concept>
    <genre>{genre}</genre>
    <narrative_style>{narrative_style}</narrative_style>
    <characters>{characters}</characters>
    <themes>{themes}</themes>
    <complexity>{complexity_level}</complexity>
    {twist_instruction}
    <additional_instructions>{additional_instructions}</additional_instructions>
    
    For each chapter/scene:
    1. Include which characters appear
    2. Describe the emotional tone and purpose
    3. Show how this advances the plot or develops characters
    4. Maintain consistent character motivations
    
    Follow the dramatic arc stages of exposition → inciting incident → rising action → midpoint → complications → climax → resolution.
    """
    
    completion = groq_provider.chat.completions.create(
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
        temperature=0.7,  # Higher for creative structure
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