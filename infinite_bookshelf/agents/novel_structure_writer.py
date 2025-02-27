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
    groq_provider,
    narrative_arc: str = "auto",
    language: str = "English"  # Add language parameter
):
    """
    Generate a structured novel outline with proper dramaturgy.
    Returns novel structure in JSON format with chapters and key scenes.
    
    Parameters:
        narrative_arc: One of ["rags_to_riches", "riches_to_rags", "man_in_hole", 
                              "icarus", "cinderella", "oedipus", "auto"]
        language: The language for the generated content
    """
    # Narrative arc descriptions
    arc_descriptions = {
        "rags_to_riches": "A continuous upward progression (rise) from hardship to success",
        "riches_to_rags": "A steady decline (fall) from success or stability to failure",
        "man_in_hole": "A descent into trouble followed by recovery (fall then rise)",
        "icarus": "An initial rise that ultimately leads to a downfall (rise then fall)",
        "cinderella": "An uplifting rise, a setback, then final recovery (rise-fall-rise)",
        "oedipus": "A decline, a momentary recovery, then final downfall (fall-rise-fall)"
    }
    
    # Select appropriate arc description
    if narrative_arc == "auto":
        arc_instruction = "Choose the most appropriate narrative arc for this genre and concept."
    else:
        arc_instruction = f"Use the '{narrative_arc}' narrative arc: {arc_descriptions.get(narrative_arc, 'Custom arc')}."
    
    # Create a prompt that incorporates dramaturgical principles
    twist_instruction = "Include a surprising plot twist" if has_twist else ""
    
    # Add language instruction to the system prompt
    language_instruction = f"Generate all chapter titles and descriptions in {language}."
    
    SYSTEM_PROMPT = f"""
    You are an expert novel structure designer. Create a compelling novel structure in JSON format following classical dramaturgy and narrative arcs.
    
    {language_instruction}
    
    IMPORTANT: Structure your output with ACTUAL CHAPTER TITLES, not metadata fields.
    For each chapter or scene, include a "dramaturgy_level" (scale 1-10) that indicates 
    the emotional intensity, tension, or dramatic impact of that section.
    
    Follow this format:
    {{{{
      "Chapter 1: [Descriptive Title]": {{{{
        "description": "Description of chapter content...",
        "dramaturgy_level": 2,
        "setting_focus": true/false,
        "character_focus": true/false
      }}}},
      "Chapter 2: [Descriptive Title]": {{{{
        "description": "Description of chapter content...",
        "dramaturgy_level": 3,
        "setting_focus": true/false,
        "character_focus": true/false,
        "scenes": {{{{
          "Scene 1: [Title]": {{{{
            "description": "Description of scene content...",
            "dramaturgy_level": 4,
            "setting_focus": true/false,
            "character_focus": true/false
          }}}}
        }}}}
      }}}}
    }}}}
    
    The dramaturgy levels should follow the chosen narrative arc, with:
    - Lower levels (1-3) for introductions, exposition, or falling action
    - Medium levels (4-6) for rising action or complications
    - Higher levels (7-10) for climactic moments, major revelations, or intense confrontations
    
    Set "setting_focus" to true for sections that should emphasize vivid location descriptions.
    Set "character_focus" to true for sections that should emphasize character development/descriptions.
    
    DO NOT use structural terms like "EXPOSITION", "RISING ACTION", etc. as chapter titles.
    DO NOT include metadata fields like "narrative_arc", "emotional_tone", "characters_involved" as keys.
    """
    
    USER_PROMPT = f"""
    Create a novel structure with the following parameters:
    
    <concept>{prompt}</concept>
    <characters>{characters}</characters>
    <genre>{genre}</genre>
    <narrative_style>{narrative_style}</narrative_style>
    <themes>{themes}</themes>
    <complexity>{complexity_level}</complexity>
    <narrative_arc_instruction>{arc_instruction}</narrative_arc_instruction>
    {twist_instruction}
    <additional_instructions>{additional_instructions}</additional_instructions>
    
    Create chapter titles and descriptions that:
    1. Sound like actual book chapter titles (e.g., "Chapter 1: The Awakening")
    2. Follow a natural progression through the chosen narrative arc
    3. Include a dramaturgy_level (1-10) for each chapter/scene
    4. Specify which sections should focus on setting descriptions
    5. Specify which sections should focus on character descriptions
    
    Structure your novel with 10-15 chapters with engaging, descriptive titles.
    Each chapter should advance the story through the dramatic arc stages.
    """
    
    completion_params = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 8000,
        "top_p": 1,
        "stream": False,
        "response_format": {"type": "json_object"},
        "stop": None,
    }

    # Add reasoning_format if using DeepSeek model
    if "deepseek" in model.lower():
        completion_params["reasoning_format"] = "hidden"

    completion = groq_provider.chat.completions.create(**completion_params)

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