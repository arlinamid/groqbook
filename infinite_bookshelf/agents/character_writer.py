"""
Agent to generate character profiles for the novel
"""

import json
from ..inference import GenerationStatistics

def generate_characters(
    prompt: str, 
    additional_instructions: str,
    number_of_characters: int, 
    model: str, 
    groq_provider,
    language: str = "English"
):
    """
    Generate detailed character profiles based on the novel concept.
    Returns character profiles in JSON format and generation statistics.
    """
    
    # Add language instruction to the system prompt
    language_instruction = f"Generate all character descriptions in {language}."
    
    # Create a clear example of the expected JSON structure
    example_character = {
        "Character_Name": {
            "role": "Protagonist/Antagonist/Supporting",
            "personality": "Key personality traits",
            "appearance": "Physical description",
            "speaking_style": "Vocal patterns and language use",
            "motivations": "What drives this character",
            "conflicts": "Internal and external struggles",
            "backstory": "Relevant history and background"
        }
    }
    
    # Convert example to JSON string with proper formatting
    example_json = json.dumps(example_character, indent=2, ensure_ascii=False)
    
    SYSTEM_PROMPT = f"""
    You are an expert character designer. Create {number_of_characters} detailed fictional characters in VALID JSON format.
    
    {language_instruction}
    
    Return ONLY a JSON object with character names as keys and their details as nested objects.
    Follow this exact structure:
    
    {example_json}
    
    IMPORTANT:
    - Ensure all JSON is properly formatted with double quotes for keys and string values
    - Do not include any explanatory text outside the JSON structure
    - Do not include any trailing commas
    - Create exactly {number_of_characters} characters
    
    Make each character:
    1. Psychologically complex with realistic strengths and flaws
    2. Have distinct personality, appearance, and speaking style
    3. Possess clear motivations, conflicts, and backstory
    4. Fit the genre and tone of the story concept
    """

    USER_PROMPT = f"""
    Create {number_of_characters} detailed and complex character profiles in VALID JSON format for a novel with the following concept:

    <concept>{prompt}</concept>

    <additional_instructions>{additional_instructions}</additional_instructions>
    
    Your response must be valid JSON that I can parse programmatically. 
    Include only the JSON object, no additional text.
    """

    # Create completion parameters dictionary
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
        "max_tokens": 4000,
        "top_p": 1,
        "stream": False,
        "response_format": {"type": "json_object"},
        "stop": None,
    }
    
    # Add reasoning_format if using DeepSeek model
    if "deepseek" in model.lower():
        completion_params["reasoning_format"] = "hidden"
    
    # Rate limit handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = groq_provider.chat.completions.create(**completion_params)
            
            # Verify the response is valid JSON
            response_content = completion.choices[0].message.content
            json.loads(response_content)  # This will throw an error if invalid JSON
            
            usage = completion.usage
            statistics = GenerationStatistics(
                input_time=usage.prompt_time,
                output_time=usage.completion_time,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_time=usage.total_time,
                model_name=model,
            )

            return statistics, response_content
            
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                # If all retries fail, create a basic JSON structure manually
                fallback_characters = {}
                for i in range(1, number_of_characters+1):
                    fallback_characters[f"Character {i}"] = {
                        "role": "Please edit this character",
                        "personality": "Generated character had invalid JSON format",
                        "appearance": "Please add description",
                        "speaking_style": "Please add speaking style",
                        "motivations": "Please add motivations",
                        "conflicts": "Please add conflicts",
                        "backstory": "Please add backstory"
                    }
                
                fallback_json = json.dumps(fallback_characters, ensure_ascii=False)
                
                # Create minimal statistics
                fallback_stats = GenerationStatistics(
                    input_time=0,
                    output_time=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_time=0,
                    model_name=model,
                )
                
                return fallback_stats, fallback_json
        
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            if attempt == max_retries - 1:
                raise 