"""
Agent to generate novel section content with narrative depth
"""

import time
import random
import streamlit as st
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
    groq_provider,
    dramaturgy_level: int = 5,
    setting_focus: bool = False,
    character_focus: bool = False,
    continuity_text: str = "",
    language: str = "English"
):
    """
    Generate immersive, narratively consistent novel content.
    Maintains character consistency and follows given tone/style.
    
    Parameters:
        dramaturgy_level: Intensity level (1-10) for this section
        setting_focus: Whether to emphasize setting descriptions
        character_focus: Whether to emphasize character descriptions
        continuity_text: Last few sentences from previous section
        language: The language for the generated content
    """
    
    # Add language instruction to the system prompt
    language_instruction = f"Write all narrative content in {language}."
    
    # Add dramaturgy instruction based on level
    dramaturgy_instruction = ""
    if dramaturgy_level <= 3:
        dramaturgy_instruction = f"""
        This is a LOW INTENSITY (level {dramaturgy_level}/10) section. Focus on:
        - Slower pacing, more introspection and descriptive passages
        - Character development and backstory exposition
        - Setting establishment and worldbuilding
        - Subtle foreshadowing and quiet moments of reflection
        """
    elif dramaturgy_level <= 6:
        dramaturgy_instruction = f"""
        This is a MEDIUM INTENSITY (level {dramaturgy_level}/10) section. Balance:
        - Moving the plot forward while developing characters
        - Building tension through obstacles and complications
        - Revealing important information through dialogue and events
        - Creating an engaging rhythm of action and reflection
        """
    else:
        dramaturgy_instruction = f"""
        This is a HIGH INTENSITY (level {dramaturgy_level}/10) section. Emphasize:
        - Fast-paced narration with shorter, punchier sentences
        - High stakes and immediate dramatic tension
        - Impactful dialogue and decisive action
        - Emotional peaks, revelations, or confrontations
        - Vivid sensory details during key moments
        """
    
    # Add setting and character focus instructions
    setting_instruction = ""
    if setting_focus:
        setting_instruction = """
        SETTING FOCUS: This section should include VIVID LOCATION DESCRIPTIONS:
        - Establish a strong sense of place with rich sensory details
        - Describe the atmosphere, lighting, sounds, smells, and textures
        - Show how the environment affects the characters' emotions
        - Use the setting to reinforce the mood and themes of this section
        """
    
    character_instruction = ""
    if character_focus:
        character_instruction = """
        CHARACTER FOCUS: This section should include DETAILED CHARACTER DESCRIPTIONS:
        - Describe physical appearance, mannerisms, clothing, and body language
        - Reveal character thoughts, emotions, and reactions in depth
        - Show how characters relate to each other through dialogue and interactions
        - Highlight character growth, changes, or important realizations
        """
    
    SYSTEM_PROMPT = f"""
    You are an expert fiction writer. Write compelling, emotionally resonant narrative content that:
    
    {language_instruction}
    
    1. Shows rather than tells when describing characters and settings
    2. Uses natural-sounding dialogue appropriate to each character
    3. Balances description, dialogue, and action
    4. Creates appropriate pacing for the scene's emotional tone
    5. Maintains consistent character voices and behaviors
    6. Advances both plot and character development
    7. Creates smooth transitions between sections of the story
    
    {dramaturgy_instruction}
    {setting_instruction}
    {character_instruction}
    
    Only output the narrative content, without additional explanations.
    """
    
    # Add continuity instructions if provided
    continuity_instruction = ""
    if continuity_text:
        continuity_instruction = f"""
        <continuity>
        The previous section ended with: "{continuity_text}"
        Ensure your narrative flows smoothly from this point, maintaining consistent tone and context.
        </continuity>
        """
    
    USER_PROMPT = f"""
    Write an engaging narrative section with the following parameters:
    
    <section_title>{title}</section_title>
    <section_description>{section_description}</section_description>
    
    <genre>{genre}</genre>
    <tone>{tone}</tone>
    <narrative_style>{narrative_style}</narrative_style>
    
    <plot_context>{plot_context}</plot_context>
    <characters>{characters}</characters>
    
    <previous_content_summary>{previous_sections_summary}</previous_content_summary>
    {continuity_instruction}
    
    <additional_instructions>{additional_instructions}</additional_instructions>
    
    Create immersive content that advances the story while developing characters. 
    Balance dialogue, action, and description.
    Maintain consistent characterization with previously established traits.
    """
    
    # Create stream parameters dictionary
    stream_params = {
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
        "temperature": 0.8,  # Higher for creative fiction
        "max_tokens": 8000,
        "top_p": 1,
        "stream": True,
        "stop": None,
    }
    
    # Add reasoning_format if using DeepSeek model
    if "deepseek" in model.lower():
        stream_params["reasoning_format"] = "hidden"
    
    # Rate limit handling with exponential backoff
    max_retries = 5
    base_delay = 1  # Start with a 1-second delay
    
    for attempt in range(max_retries):
        try:
            stream = groq_provider.chat.completions.create(**stream_params)
            
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
            
            # Successfully completed streaming, exit the retry loop
            break
            
        except Exception as e:
            error_message = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_message:
                # Extract retry-after if available
                retry_after = None
                if hasattr(e, 'headers') and 'retry-after' in e.headers:
                    retry_after = int(e.headers['retry-after'])
                
                # Calculate wait time with exponential backoff
                if retry_after:
                    wait_time = retry_after
                else:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) * base_delay + random.uniform(0, 1)
                
                # Update the user through streamlit
                yield f"\n[Rate limit reached. Waiting {wait_time:.1f} seconds to retry...]\n"
                
                # Wait before retrying
                time.sleep(wait_time)
                
                # Continue to next retry attempt
                continue
            
            # If it's not a rate limit error or we've exceeded max retries
            if attempt == max_retries - 1:
                yield f"\n[Error generating content: {error_message}]\n"
                raise 