"""
Agent to generate character profiles for the novel
"""

from ..inference import GenerationStatistics

def generate_characters(
    prompt: str, 
    additional_instructions: str,
    number_of_characters: int, 
    model: str, 
    groq_provider
):
    """
    Generate detailed character profiles based on the novel concept.
    Returns character profiles in JSON format and generation statistics.
    """
    completion = groq_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Create detailed character profiles for a novel in JSON format. Include name, age, physical appearance, personality traits, background, motivations, fears, desires, and relationships with other characters.",
            },
            {
                "role": "user",
                "content": f"Create {number_of_characters} detailed and complex character profiles in JSON format for a novel with the following concept:\n\n<concept>{prompt}</concept>\n\n<additional_instructions>{additional_instructions}</additional_instructions>",
            },
        ],
        temperature=0.7,  # Higher temperature for more creative characters
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