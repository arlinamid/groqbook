# 1: Import libraries
import streamlit as st
from groq import Groq
import json

from infinite_bookshelf.agents import (
    generate_section,
    generate_book_title,
    generate_characters,
    generate_plot_structure
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
    "plot": {},
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
        plot_agent_model,
        section_agent_model,
    ) = render_novel_form(
        on_submit=disable,
        button_disabled=st.session_state.button_disabled,
        button_text=st.session_state.button_text,
    )

    # Build the enhanced prompt with all novel parameters
    if submitted:
        if len(concept_text) < 10:
            raise ValueError("Novel concept must be at least 10 characters long")

        st.session_state.button_disabled = True
        st.session_state.statistics_text = "Generating novel components in background..."
        st.session_state.generation_stage = "characters"

        placeholder = st.empty()
        display_statistics(
            placeholder=placeholder, statistics_text=st.session_state.statistics_text
        )

        if not GROQ_API_KEY:
            st.session_state.groq = Groq(api_key=groq_input_key)

        # Generate enhanced prompt with all parameters
        plot_parameters = f"""
        Genre: {genre}
        Narrative Style: {narrative_style}
        Tone: {tone}
        Plot Complexity: {complexity}
        Pacing: {pacing}
        Include Romance Subplot: {'Yes' if has_romance else 'No'}
        Include Plot Twist: {'Yes' if has_twist else 'No'}
        """
        
        combined_instructions = f"{additional_instructions}\n{plot_parameters}"
            
        # Step 1: Generate characters
        st.session_state.statistics_text = "Generating characters..."
        display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
        
        character_stats, characters_json = generate_characters(
            prompt=concept_text,
            additional_instructions=f"{combined_instructions}\n{character_seeds}",
            number_of_characters=num_characters,
            model=character_agent_model,
            groq_provider=st.session_state.groq,
        )
        
        characters_data = json.loads(characters_json)
        st.session_state.characters = characters_data
        
        # Step 2: Generate novel title
        st.session_state.statistics_text = "Generating novel title..."
        display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
        
        st.session_state.novel_title = generate_book_title(
            prompt=f"{concept_text}\nGenre: {genre}\nTone: {tone}",
            model=title_agent_model,
            groq_provider=st.session_state.groq,
        )
        
        st.write(f"## {st.session_state.novel_title}")
        
        # Step 3: Generate plot structure
        st.session_state.statistics_text = "Creating plot structure..."
        display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
        
        plot_stats, plot_json = generate_plot_structure(
            prompt=concept_text,
            characters=json.dumps(characters_data),
            genre=genre,
            narrative_style=narrative_style,
            additional_instructions=combined_instructions,
            model=plot_agent_model,
            groq_provider=st.session_state.groq,
        )
        
        # Step 4: Generate novel content
        try:
            plot_structure = json.loads(plot_json)
            st.session_state.plot = plot_structure
            
            book = Book(st.session_state.novel_title, plot_structure)
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
            print(json.dumps(plot_structure, indent=2))
            book.display_structure()
            
            # Function to generate content for each section
            def stream_section_content(sections, context=""):
                for title, content in sections.items():
                    # Create the plot context for this section
                    if isinstance(content, str):
                        section_context = f"{context}\nCurrent section: {title} - {content}"
                        
                        st.session_state.statistics_text = f"Generating section: {title}"
                        display_statistics(placeholder=placeholder, statistics_text=st.session_state.statistics_text)
                        
                        content_stream = generate_section(
                            prompt=title,
                            plot_context=section_context,
                            characters=json.dumps(characters_data),
                            tone=tone,
                            additional_instructions=combined_instructions,
                            model=section_agent_model,
                            groq_provider=st.session_state.groq,
                        )
                        
                        for chunk in content_stream:
                            # Check if GenerationStatistics data is returned
                            if isinstance(chunk, GenerationStatistics):
                                st.session_state.statistics_text = str(chunk)
                                display_statistics(
                                    placeholder=placeholder,
                                    statistics_text=st.session_state.statistics_text,
                                )
                            elif chunk:
                                st.session_state.book.update_content(title, chunk)
                    
                    elif isinstance(content, dict):
                        new_context = f"{context}\nParent section: {title}"
                        stream_section_content(content, new_context)
            
            # Start the content generation process
            stream_section_content(plot_structure)
            
        except json.JSONDecodeError:
            st.error("Failed to decode the plot structure. Please try again.")

except Exception as e:
    st.session_state.button_disabled = False
    st.error(f"An error occurred: {e}")
    
    if st.button("Clear"):
        st.rerun() 