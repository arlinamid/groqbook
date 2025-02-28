# 1: Import libraries
import streamlit as st
from groq import Groq
import json

from infinite_bookshelf.agents import (
    generate_section,
    generate_book_title,
    generate_characters,
    generate_plot_structure,
    generate_novel_structure,
    generate_novel_section,
    update_character_arcs
)
from infinite_bookshelf.inference import GenerationStatistics
from infinite_bookshelf.tools import create_markdown_file, create_pdf_file
from infinite_bookshelf.ui.components import (
    display_statistics,
    render_download_buttons,
)
from infinite_bookshelf.ui.components.novel_form import render_novel_form
from infinite_bookshelf.ui import Book, load_return_env, ensure_states


# 2: Initialize env variables and session states
GROQ_API_KEY = load_return_env(["GROQ_API_KEY"])["GROQ_API_KEY"]

states = {
    "api_key": GROQ_API_KEY,
    "button_disabled": False,
    "button_text": "Generate Novel",
    "statistics_text": "",
    "novel_title": "",
    "characters": {},
    "novel_structure": {},
    "character_arcs": {},
    "completed_sections": "",
    "generation_stage": "init"
}

# Initialize the Groq client regardless, but with a check before use
states["groq"] = None if not GROQ_API_KEY else Groq(api_key=GROQ_API_KEY)

ensure_states(states)

# Validate API key is available
if not st.session_state.groq:
    st.error("⚠️ Groq API key not found. Please set your GROQ_API_KEY in .env file or environment variables.")
    st.stop()


# 3: Define Streamlit page structure and functionality
st.write(
    """
# AI Novel Generator
Create rich, narrative-driven stories with dynamic characters and engaging plots
"""
)

def disable():
    st.session_state.button_disabled = True

def enable():
    st.session_state.button_disabled = False

try:
    if st.button("End Generation and Download Novel"):
        if "book" in st.session_state:
            render_download_buttons(st.session_state.get("book"))

    # Render the novel generation form
    (
        submitted,
        groq_input_key,
        concept_text,
        genre,
        narrative_style,
        tone,
        num_characters,
        has_romance,
        has_twist,
        complexity,
        pacing,
        additional_instructions,
        character_seeds,
        narrative_arc,
        language,
        title_agent_model,
        character_agent_model,
        plot_agent_model,
        section_agent_model,
    ) = render_novel_form(
        on_submit=disable,
        button_disabled=st.session_state.button_disabled,
        button_text=st.session_state.button_text,
    )

    if submitted:
        try:
            # If the user provided an API key in the form, update it
            if groq_input_key:
                st.session_state.api_key = groq_input_key
                st.session_state.groq = Groq(api_key=groq_input_key)
            
            # Verify we have a Groq client initialized
            if not st.session_state.groq:
                st.error("⚠️ Groq API key not found. Please provide a valid Groq API key.")
                st.session_state.button_disabled = False
                st.stop()
                
            # Create a placeholder for displaying generation statistics
            placeholder = st.empty()
            total_generation_statistics = GenerationStatistics(
                input_time=0, 
                output_time=0, 
                input_tokens=0, 
                output_tokens=0, 
                total_time=0, 
                model_name="combined"  # Using "combined" as it will track multiple models
            )
            
            # 1. GENERATE CHARACTERS
            st.session_state.statistics_text = "Creating characters..."
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            # Prepare character generation prompt with seeds if provided
            combined_character_instructions = additional_instructions
            if character_seeds:
                combined_character_instructions += f"\nCharacter seeds: {character_seeds}"
            
            char_stats, characters_json = generate_characters(
                prompt=concept_text,
                additional_instructions=combined_character_instructions,
                number_of_characters=num_characters,
                model=character_agent_model,
                groq_provider=st.session_state.groq,
                language=language
            )
            
            total_generation_statistics.add(char_stats)
            st.session_state.statistics_text = str(total_generation_statistics)
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            characters_data = json.loads(characters_json)
            st.session_state.characters = characters_data
            
            # Store character goals for later use in character arc tracking
            character_goals = {}
            for name, info in characters_data.items():
                if isinstance(info, dict) and "motivations" in info:
                    character_goals[name] = info["motivations"]
                elif isinstance(info, dict) and "goals" in info:
                    character_goals[name] = info["goals"]
                else:
                    character_goals[name] = "Unknown goals"
            
            # 2. GENERATE PLOT STRUCTURE - with narrative arc
            st.session_state.statistics_text = "Creating plot structure..."
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            # Create romance instruction if needed
            romance_instruction = "Include a romance subplot" if has_romance else ""
            combined_instructions = f"{additional_instructions}\n{romance_instruction}".strip()
            
            plot_stats, plot_structure_json = generate_plot_structure(
                prompt=concept_text,
                characters=characters_json,
                genre=genre,
                narrative_style=narrative_style,
                additional_instructions=combined_instructions,
                model=plot_agent_model,
                groq_provider=st.session_state.groq,
                narrative_arc=narrative_arc,
                language=language
            )
            
            total_generation_statistics.add(plot_stats)
            st.session_state.statistics_text = str(total_generation_statistics)
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            plot_structure = json.loads(plot_structure_json)
            st.session_state.plot_structure = plot_structure
            
            # 3. GENERATE NOVEL STRUCTURE - based on plot structure
            st.session_state.statistics_text = "Creating detailed novel structure..."
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            # Extract themes from concept
            themes = concept_text.split()[:5]  # Simple extraction of potential themes
            themes_str = ", ".join(themes)
            
            # Pass plot structure as part of instructions
            structure_instructions = f"{combined_instructions}\nFollow this plot structure: {json.dumps(plot_structure)}"
            
            structure_stats, novel_structure_json = generate_novel_structure(
                prompt=concept_text,
                characters=characters_json,
                genre=genre,
                narrative_style=narrative_style,
                themes=themes_str,
                has_twist=has_twist,
                complexity_level=complexity,
                additional_instructions=structure_instructions,
                narrative_arc=narrative_arc,
                model=plot_agent_model,
                groq_provider=st.session_state.groq,
                language=language
            )
            
            total_generation_statistics.add(structure_stats)
            st.session_state.statistics_text = str(total_generation_statistics)
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            # Filter out metadata keys from the structure
            metadata_fields = [
                "narrative_arc", "emotional_tone", "characters_involved", 
                "plot_structure", "narrative_advancement", "exposition", 
                "inciting_incident", "rising_action", "midpoint", 
                "complications", "climax", "resolution"
            ]

            def filter_structure(structure):
                filtered = {}
                for key, value in structure.items():
                    key_lower = key.lower()
                    if not any(field.lower() in key_lower for field in metadata_fields):
                        if isinstance(value, dict):
                            filtered[key] = filter_structure(value)
                        else:
                            filtered[key] = value
                return filtered

            novel_structure = filter_structure(json.loads(novel_structure_json))
            
            st.session_state.novel_structure = novel_structure
            
            # Create the book object to populate with content
            title = generate_book_title(concept_text, title_agent_model, st.session_state.groq)
            st.session_state.novel_title = title
            
            book = Book(title, novel_structure)
            st.session_state.book = book
            
            # Display characters and structure
            st.markdown("## Characters")
            char_cols = st.columns(2)
            for i, (name, info) in enumerate(characters_data.items()):
                with char_cols[i % 2]:
                    st.markdown(f"### {name}")
                    if isinstance(info, dict):
                        for key, value in info.items():
                            st.markdown(f"**{key}**: {value}")
                    else:
                        st.markdown(info)
            
            st.markdown("---")
            
            # Print the book structure for debugging
            print(json.dumps(novel_structure, indent=2))
            book.display_structure()
            
            # 4. GENERATE SECTIONS with continuity tracking
            # Find the stream_section_content function and update it:
            
            # Function to generate content for each section with character arc tracking
            def stream_section_content(structure, summary="", context="", section_depth=0):
                last_4_sentences = ""  # Track last 4 sentences for continuity
                
                for title, content in structure.items():
                    # Display title
                    section_placeholder = st.empty()
                    section_placeholder.markdown(f"### {title}")
                    
                    # Extract dramaturgy parameters if present
                    dramaturgy_level = 5  # Default level
                    setting_focus = False
                    character_focus = False
                    
                    section_description = ""
                    
                    if isinstance(content, dict):
                        # Check if this is a content dictionary with our metadata fields
                        if "description" in content:
                            section_description = content["description"]
                            # Extract dramaturgy parameters
                            dramaturgy_level = content.get("dramaturgy_level", 5)
                            setting_focus = content.get("setting_focus", False)
                            character_focus = content.get("character_focus", False)
                            
                            # Create a readable description of this section's focus for the user
                            focus_text = ""
                            if setting_focus:
                                focus_text += "📍 Setting Focus"
                            if character_focus:
                                focus_text += " 👤 Character Focus"
                            if focus_text:
                                section_placeholder.markdown(f"### {title} {focus_text}")
                                
                            # Display dramaturgy level as a progress bar
                            dramaturgy_col1, dramaturgy_col2 = st.columns([1, 3])
                            with dramaturgy_col1:
                                st.write("Intensity:")
                            with dramaturgy_col2:
                                st.progress(dramaturgy_level/10, text=f"Level {dramaturgy_level}/10")
                            
                            try:
                                # Generate section content with dramaturgy parameters
                                section_text = ""
                                for token in generate_novel_section(
                                    title=title,
                                    section_description=section_description,
                                    plot_context=context,
                                    characters=json.dumps(st.session_state.characters),
                                    genre=genre,
                                    tone=tone,
                                    narrative_style=narrative_style,
                                    previous_sections_summary=summary,
                                    continuity_text=last_4_sentences,
                                    dramaturgy_level=dramaturgy_level,
                                    setting_focus=setting_focus,
                                    character_focus=character_focus,
                                    additional_instructions=additional_instructions,
                                    model=section_agent_model,
                                    groq_provider=st.session_state.groq,
                                    language=language
                                ):
                                    if isinstance(token, str):
                                        section_text += token
                                        section_placeholder.markdown(f"### {title}\n\n{section_text}")
                                    elif isinstance(token, GenerationStatistics):
                                        total_generation_statistics.add(token)
                                        st.session_state.statistics_text = str(total_generation_statistics)
                                        display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
                                
                                # Skip metadata fields that aren't actual sections
                                metadata_fields = [
                                    "narrative_arc", "emotional_tone", "characters_involved", 
                                    "plot_structure", "narrative_advancement", "EXPOSITION", 
                                    "INCITING INCIDENT", "RISING ACTION", "MIDPOINT", 
                                    "COMPLICATIONS", "CLIMAX", "RESOLUTION"
                                ]

                                if not any(title.lower() == field.lower() for field in metadata_fields):
                                    try:
                                        book.add_section_content(title, section_text)
                                    except AttributeError:
                                        # If the method doesn't exist, just store the content in a dictionary
                                        if not hasattr(book, 'content'):
                                            book.content = {}
                                        book.content[title] = section_text
                                        print(f"Stored content for '{title}' in book.content dictionary")
                                
                                # Update summary with new section
                                if summary:
                                    summary += f"\n\n{title}: {section_text[:300]}..."
                                else:
                                    summary = f"{title}: {section_text[:300]}..."
                                
                                # Extract last 4 sentences for continuity in next section
                                sentences = section_text.split(".")
                                last_4_sentences = ".".join(sentences[-5:-1]) + "." if len(sentences) > 4 else section_text
                                
                                # Update completed sections for character arc tracking
                                st.session_state.completed_sections += f"\n\n{title}: {section_text}"
                                
                                # Update character arcs after significant sections
                                if section_depth <= 1:  # Only update for main chapters or key scenes
                                    try:
                                        arc_stats, updated_character_arcs = update_character_arcs(
                                            characters=json.dumps(st.session_state.characters),
                                            current_plot_point=f"{title}: {content}",
                                            completed_sections=st.session_state.completed_sections[-5000:],  # Last 5000 chars
                                            character_goals=json.dumps(character_goals),
                                            model=character_agent_model,
                                            groq_provider=st.session_state.groq,
                                            narrative_arc=narrative_arc,
                                            language=language
                                        )
                                        
                                        st.session_state.characters = json.loads(updated_character_arcs)
                                        st.session_state.character_arcs = st.session_state.characters
                                        
                                        total_generation_statistics.add(arc_stats)
                                        st.session_state.statistics_text = str(total_generation_statistics)
                                        display_statistics(
                                            placeholder=placeholder, 
                                            statistics_text=st.session_state.statistics_text
                                        )
                                    except Exception as e:
                                        print(f"Error updating character arcs: {e}")
                            
                            except Exception as e:
                                error_message = str(e)
                                if "429" in error_message or "rate_limit" in error_message.lower():
                                    st.warning("⚠️ **Rate limit reached**: Slowing down generation to stay within API limits. This may take longer than usual.")
                                    # Retry logic handled by our rate limiter modules
                                else:
                                    st.error(f"Error: {error_message}")
                                section_text = f"[Error generating content for section: {str(e)}]"
                                # Continue with basic handling to prevent breaking the app flow
                            
                            # For nested sections, pass context and continue generation
                            if isinstance(content, dict):
                                new_context = f"{context}\nParent section: {title}"
                                summary = stream_section_content(content, summary, new_context, section_depth + 1)
                
                return summary
            
            # Start the content generation process
            completed_sections_summary = stream_section_content(novel_structure)
            
            # Generation complete
            st.session_state.button_disabled = False
            st.session_state.button_text = "Regenerate Novel"
            
            # Show updated character arcs at the end
            if st.session_state.get("character_arcs"):
                st.markdown("## Character Development Throughout the Story")
                for character, details in st.session_state.character_arcs.items():
                    with st.expander(f"{character}'s Arc"):
                        if isinstance(details, dict):
                            for key, value in details.items():
                                st.markdown(f"**{key}**: {value}")
                        else:
                            st.markdown(details)
            
        except json.JSONDecodeError:
            st.error("Failed to decode JSON data. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.button_disabled = False

except Exception as e:
    st.session_state.button_disabled = False
    st.error(f"An error occurred: {e}")
    
    if st.button("Clear"):
        st.rerun() 