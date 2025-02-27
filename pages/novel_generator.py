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

if GROQ_API_KEY:
    states["groq"] = Groq()  

ensure_states(states)


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
        title_agent_model,
        character_agent_model,
        structure_agent_model,
        section_agent_model,
    ) = render_novel_form(
        on_submit=disable,
        button_disabled=st.session_state.button_disabled,
        button_text=st.session_state.button_text,
    )

    if submitted:
        try:
            # Create a placeholder for displaying generation statistics
            placeholder = st.empty()
            total_generation_statistics = GenerationStatistics(model_name="various")
            
            # Store API key if provided
            if groq_input_key:
                st.session_state.api_key = groq_input_key
                st.session_state.groq = Groq(api_key=groq_input_key)
                
            # Set up additional instructions
            romance_instruction = "Include a romantic subplot between characters." if has_romance else ""
            pacing_instruction = f"The story should have {pacing.lower()} pacing."
            complexity_instruction = f"Create a {complexity.lower()}-complexity plot structure."
            
            combined_instructions = f"{additional_instructions}\n{romance_instruction}\n{pacing_instruction}\n{complexity_instruction}"
            
            if character_seeds:
                combined_instructions += f"\nUse these character ideas: {character_seeds}"
                
            # Generate novel title
            st.session_state.statistics_text = "Generating title..."
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            title = generate_book_title(
                prompt=concept_text,
                model=title_agent_model,
                groq_provider=st.session_state.groq,
            )
            
            st.session_state.novel_title = title
            st.markdown(f"# {title}")
            
            # Generate characters
            st.session_state.statistics_text = "Generating characters..."
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            character_stats, characters_json = generate_characters(
                prompt=concept_text,
                additional_instructions=combined_instructions,
                number_of_characters=num_characters,
                model=character_agent_model,
                groq_provider=st.session_state.groq,
            )
            
            total_generation_statistics.add(character_stats)
            st.session_state.statistics_text = str(total_generation_statistics)
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            characters_data = json.loads(characters_json)
            st.session_state.characters = characters_data
            
            # Extract character goals for the arc tracker
            character_goals = {}
            for name, info in characters_data.items():
                if isinstance(info, dict) and "motivations" in info:
                    character_goals[name] = info["motivations"]
                elif isinstance(info, dict) and "goals" in info:
                    character_goals[name] = info["goals"]
                else:
                    character_goals[name] = "Unknown goals"
            
            # Generate novel structure using the enhanced structure writer
            st.session_state.statistics_text = "Creating story structure..."
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            # Extract themes from concept
            themes = concept_text.split()[:5]  # Simple extraction of potential themes
            themes_str = ", ".join(themes)
            
            structure_stats, novel_structure_json = generate_novel_structure(
                prompt=concept_text,
                characters=characters_json,
                genre=genre,
                narrative_style=narrative_style,
                themes=themes_str,
                has_twist=has_twist,
                complexity_level=complexity,
                additional_instructions=combined_instructions,
                model=structure_agent_model,
                groq_provider=st.session_state.groq,
            )
            
            total_generation_statistics.add(structure_stats)
            st.session_state.statistics_text = str(total_generation_statistics)
            display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
            
            novel_structure = json.loads(novel_structure_json)
            st.session_state.novel_structure = novel_structure
            
            # Create the book object to populate with content
            book = Book(title, novel_structure)
            st.session_state.book = book
            
            # Display character information
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
            
            # Function to generate content for each section with character arc tracking
            def stream_section_content(sections, summary="", context="", section_depth=0):
                for title, content in sections.items():
                    # Create the plot context for this section
                    if isinstance(content, str):
                        section_context = f"{context}\nCurrent section: {title} - {content}"
                        
                        st.session_state.statistics_text = f"Generating section: {title}"
                        display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
                        
                        # Use novel_section_writer instead of regular section_writer
                        content_stream = generate_novel_section(
                            title=title,
                            section_description=content,
                            plot_context=section_context,
                            characters=json.dumps(st.session_state.get("character_arcs", characters_data)),
                            genre=genre,
                            tone=tone,
                            narrative_style=narrative_style,
                            previous_sections_summary=summary,
                            additional_instructions=combined_instructions,
                            model=section_agent_model,
                            groq_provider=st.session_state.groq,
                        )
                        
                        section_content = ""
                        for chunk in content_stream:
                            # Check if GenerationStatistics data is returned
                            if isinstance(chunk, GenerationStatistics):
                                st.session_state.statistics_text = str(chunk)
                                display_statistics(
                                    placeholder=placeholder,
                                    statistics_text=st.session_state.statistics_text,
                                )
                            elif chunk:
                                section_content += chunk
                                st.session_state.book.update_content(title, chunk)
                        
                        # Add this section to the completed sections summary (abbreviated)
                        section_summary = f"{title}: {section_content[:200]}..."
                        summary += section_summary + "\n\n"
                        
                        # Update character arcs after significant plot points (chapter level or major scene)
                        if section_depth <= 1 and len(section_content) > 500:  # Only for chapters or major scenes
                            try:
                                arc_stats, updated_arcs = update_character_arcs(
                                    characters=json.dumps(st.session_state.get("character_arcs", characters_data)),
                                    current_plot_point=f"{title}: {content}",
                                    completed_sections=section_summary,
                                    character_goals=json.dumps(character_goals),
                                    model=character_agent_model,
                                    groq_provider=st.session_state.groq,
                                )
                                
                                # Update character arcs for future sections
                                st.session_state.character_arcs = json.loads(updated_arcs)
                                
                                # Add to total stats
                                total_generation_statistics.add(arc_stats)
                                st.session_state.statistics_text = str(total_generation_statistics)
                                display_statistics(
                                    placeholder=placeholder, 
                                    statistics_text=st.session_state.statistics_text
                                )
                            except Exception as e:
                                print(f"Error updating character arcs: {e}")
                    
                    elif isinstance(content, dict):
                        new_context = f"{context}\nParent section: {title}"
                        summary = stream_section_content(content, summary, new_context, section_depth + 1)
            
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