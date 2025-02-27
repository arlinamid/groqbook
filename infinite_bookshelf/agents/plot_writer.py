"""
Agent to generate novel plot structure with narrative arcs
"""

from ..inference import GenerationStatistics, groq_limiter

def generate_plot_structure(
    prompt: str,
    characters: str,
    genre: str,
    narrative_style: str,
    additional_instructions: str,
    model: str,
    groq_provider,
    narrative_arc: str = "auto",
    language: str = "English"  # Add language parameter
):
    """
    Generate a narrative arc and plot structure for the novel.
    Returns plot structure in JSON format and generation statistics.
    
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
    
    # Add language instruction to the system prompt
    language_instruction = f"Generate all plot content in {language}."
    
    USER_PROMPT = f"""
    Create a detailed plot structure for a novel with the following, and return the result in JSON format:
    
    <concept>{prompt}</concept>
    <genre>{genre}</genre>
    <narrative_style>{narrative_style}</narrative_style>
    <characters>{characters}</characters>
    <additional_instructions>{additional_instructions}</additional_instructions>
    
    <narrative_arc_instruction>{arc_instruction}</narrative_arc_instruction>
    
    Create a structure following the classical dramatic structure, but DO NOT use these terms as keys in your JSON:
    1. EXPOSITION: Introduce characters, setting, and initial situation
    2. INCITING INCIDENT: The event that sets the story in motion
    3. RISING ACTION: Escalating conflicts and complications
    4. MIDPOINT: A major revelation or shift in perspective
    5. COMPLICATIONS: Stakes rise, challenges intensify
    6. CLIMAX: The highest point of tension where the main conflict comes to a head
    7. RESOLUTION: Aftermath and tying up of loose ends
    
    Instead, create a sequence of 8-12 plot points with descriptive names that follow this structure.
    
    For each plot point, explain:
    - What happens
    - Which characters are involved
    - How this advances the narrative arc
    - The emotional tone of this section
    """
    
    SYSTEM_PROMPT = f"""
    You are a master storyteller with expertise in narrative structure.
    
    {language_instruction}
    
    Structure your response as a JSON object with actual plot points and events, not structural metadata.
    
    The six primary narrative arcs provide guidance for your plot's emotional trajectory:
    1. RAGS TO RICHES (Rise): A continuous upward progression 
    2. RICHES TO RAGS (Fall): A steady decline
    3. MAN IN A HOLE (Fall-Rise): A descent into trouble followed by recovery
    4. ICARUS (Rise-Fall): An initial rise that leads to a downfall
    5. CINDERELLA (Rise-Fall-Rise): An uplifting rise, a setback, then final recovery
    6. OEDIPUS (Fall-Rise-Fall): A decline, a momentary recovery, then final downfall

    Follow this format for your JSON output:
    {{{{
      "Plot_Point_1": "Description of what happens at this point in the story...",
      "Plot_Point_2": "Description of what happens next...",
      ...
    }}}}

    Ensure character motivations drive the plot and that the emotional trajectory follows the selected arc.
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
        "temperature": 0.6,
        "max_tokens": 8000,
        "top_p": 1,
        "stream": False,
        "response_format": {"type": "json_object"},
        "stop": None,
    }

    # Add reasoning_format if using DeepSeek model
    if "deepseek" in model.lower():
        completion_params["reasoning_format"] = "hidden"
    
    # Estimate token count for rate limiting
    # A rough estimate: 1 token ≈ 4 characters for English, adjust for other languages
    system_tokens = len(SYSTEM_PROMPT) // 4
    user_tokens = len(USER_PROMPT) // 4
    estimated_input_tokens = system_tokens + user_tokens
    estimated_output_tokens = 8000  # Max tokens allowed
    
    # Use rate limiter to ensure we don't exceed TPM limits
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Check with rate limiter before making the API call
            groq_limiter.request(estimated_input_tokens + estimated_output_tokens)
            
            # Make the API call
            completion = groq_provider.chat.completions.create(**completion_params)
            
            # Process successful response
            usage = completion.usage
            statistics = GenerationStatistics(
                input_time=usage.prompt_time,
                output_time=usage.completion_time,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_time=usage.total_time,
                model_name=model,
            )
            
            # Update rate limiter with actual token usage instead of estimate
            actual_tokens = usage.prompt_tokens + usage.completion_tokens
            groq_limiter.record_usage(actual_tokens - estimated_input_tokens - estimated_output_tokens)
            
            return statistics, completion.choices[0].message.content
            
        except Exception as e:
            error_message = str(e)
            
            # Check for rate limit errors
            if "429" in error_message or "rate_limit" in error_message.lower():
                # Extract retry-after if available
                retry_after = None
                if hasattr(e, 'headers') and 'retry-after' in e.headers:
                    retry_after = int(e.headers['retry-after'])
                
                # Notify rate limiter about the rate limit error
                groq_limiter.handle_rate_limit_error(retry_after)
                
                # Only retry if we haven't exhausted our attempts
                if attempt < max_retries - 1:
                    continue
            
            # For non-rate limit errors or if we've exhausted retries
            raise 