"""
Agent to track and maintain character arcs throughout the novel
"""

from ..inference import GenerationStatistics

def update_character_arcs(
    characters: str,
    current_plot_point: str,
    completed_sections: str,
    character_goals: str, 
    model: str,
    groq_provider
):
    """
    Tracks character development and provides updated character states
    after each significant plot point to maintain consistency.
    """
    SYSTEM_PROMPT = """
    You are a character development specialist. Track the emotional and psychological evolution 
    of characters throughout a novel, ensuring consistent character arcs in JSON format. 
    For each character, track:
    
    1. Current emotional state
    2. Relationships with other characters
    3. Knowledge gained
    4. Progress toward goals
    5. Character growth or change
    
    Return the updated character profiles in JSON format.
    """
    
    USER_PROMPT = f"""
    Update the character profiles based on the most recent developments in the story.
    Return updated character information in JSON format.
    
    <original_character_profiles>
    {characters}
    </original_character_profiles>
    
    <character_goals>
    {character_goals}
    </character_goals>
    
    <recent_plot_development>
    {current_plot_point}
    </recent_plot_development>
    
    <completed_sections_summary>
    {completed_sections}
    </completed_sections_summary>
    
    For each character, update their:
    - Current emotional state and mindset
    - Relationships with other characters (any changes)
    - Progress toward their goals
    - Knowledge or secrets they now possess
    - Character growth or regression
    
    Ensure the updates are consistent with their established personality traits.
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
        temperature=0.4,  # Lower for consistency
        max_tokens=4000,
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