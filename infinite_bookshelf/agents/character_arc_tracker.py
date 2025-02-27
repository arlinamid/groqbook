"""
Agent to track and maintain character arcs throughout the novel
"""

import time
import random
import json
from ..inference import GenerationStatistics

def update_character_arcs(
    characters: str,
    current_plot_point: str,
    completed_sections: str,
    character_goals: str,
    model: str,
    groq_provider,
    narrative_arc: str = "auto",
    language: str = "English"
):
    """Updates character development based on story progression"""
    
    # Add language instruction to system prompt
    language_instruction = f"Generate all character arc updates in {language}."
    
    # First try to parse the input characters to ensure we have valid JSON
    try:
        characters_data = json.loads(characters)
    except json.JSONDecodeError:
        print(f"Warning: Input characters JSON is not valid, attempting to continue with original string")
        characters_data = {}
    
    # Create a simplified example response format
    example_format = {
        "Character_Name": {
            "emotional_growth": "Description of how the character has grown emotionally",
            "relationship_changes": "Description of how the character's relationships have changed",
            "progress_toward_goals": "Description of progress toward character goals",
            "alignment_with_narrative_arc": "Description of how the character fits into the story arc"
        }
    }
    
    example_json = json.dumps(example_format, indent=2, ensure_ascii=False)
    
    SYSTEM_PROMPT = f"""
    You are an expert in character development and narrative arcs.
    
    {language_instruction}
    
    Track how characters evolve through a story, analyzing:
    1. Emotional growth or deterioration
    2. Relationship changes
    3. Progress toward or away from their goals
    4. Alignment with the narrative arc
    
    Return a VALID JSON object with character updates using this exact structure:
    {example_json}
    
    IMPORTANT: 
    - Make sure your response is valid JSON
    - Use double quotes for all keys and string values
    - Do not include any trailing commas
    - Do not include any special characters or formatting that would break JSON syntax
    """
    
    USER_PROMPT = f"""
    Update these character profiles based on story progress:
    
    <characters>{characters}</characters>
    <character_goals>{character_goals}</character_goals>
    <current_plot_point>{current_plot_point}</current_plot_point>
    <completed_sections>{completed_sections}</completed_sections>
    <narrative_arc>{narrative_arc}</narrative_arc>
    
    Analyze how each character is developing in relation to:
    1. Their initial goals and motivations
    2. Their relationships with other characters
    3. Their emotional and psychological state
    4. The overall narrative arc of the story
    
    Return the updated character information in VALID JSON format.
    """
    
    # Rate limit handling with exponential backoff
    max_retries = 5
    base_delay = 1  # Start with a 1-second delay
    
    for attempt in range(max_retries):
        try:
            # Create completion parameters dictionary
            completion_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                "temperature": 0.5,
                "max_tokens": 4000,
                "top_p": 1,
                "stream": False,
                "response_format": {"type": "json_object"},
                "stop": None,
            }
            
            # Add reasoning_format if using DeepSeek model
            if "deepseek" in model.lower():
                completion_params["reasoning_format"] = "hidden"
            
            completion = groq_provider.chat.completions.create(**completion_params)
            
            # Attempt to parse the JSON response to verify it's valid
            response_content = completion.choices[0].message.content
            json.loads(response_content)  # This will raise an exception if the JSON is invalid
            
            # If we got here, the JSON is valid
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
            
        except json.JSONDecodeError as json_error:
            print(f"JSON decode error on attempt {attempt+1}: {json_error}")
            # If this was our last retry, handle it differently
            if attempt == max_retries - 1:
                # Instead of failing completely, return the original characters with a minimal update
                if isinstance(characters_data, dict):
                    # Add a note about the error
                    for char_name in characters_data:
                        if isinstance(characters_data[char_name], dict):
                            characters_data[char_name]["update_status"] = f"Error: Failed to update character (JSON validation error)"
                    return GenerationStatistics(
                        input_time=0, 
                        output_time=0, 
                        input_tokens=0, 
                        output_tokens=0, 
                        total_time=0, 
                        model_name=model
                    ), json.dumps(characters_data, ensure_ascii=False)
                else:
                    # Return the original if we couldn't even parse it
                    return GenerationStatistics(
                        input_time=0, 
                        output_time=0, 
                        input_tokens=0, 
                        output_tokens=0, 
                        total_time=0, 
                        model_name=model
                    ), characters
        
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
                
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds to retry...")
                
                # Wait before retrying
                time.sleep(wait_time)
                
                # Continue to next retry attempt
                continue
            
            # If it's not a rate limit error or we've exceeded max retries
            if attempt == max_retries - 1:
                print(f"Error updating character arcs after {max_retries} attempts: {error_message}")
                # Return the original characters with minimal modifications
                if isinstance(characters_data, dict):
                    # Add a note about the error
                    for char_name in characters_data:
                        if isinstance(characters_data[char_name], dict):
                            characters_data[char_name]["update_status"] = f"Error: {error_message}"
                    return GenerationStatistics(
                        input_time=0, 
                        output_time=0, 
                        input_tokens=0, 
                        output_tokens=0, 
                        total_time=0, 
                        model_name=model
                    ), json.dumps(characters_data, ensure_ascii=False)
                else:
                    return GenerationStatistics(
                        input_time=0, 
                        output_time=0, 
                        input_tokens=0, 
                        output_tokens=0, 
                        total_time=0, 
                        model_name=model
                    ), characters 