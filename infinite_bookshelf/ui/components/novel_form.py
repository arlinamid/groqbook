"""
Component function for novel generation form
"""

import streamlit as st

MODEL_LIST = ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile", "gemma2-9b-it"]
GENRES = ["Fantasy", "Science Fiction", "Mystery", "Romance", "Thriller", "Historical Fiction", "Horror", "Adventure"]
NARRATIVE_STYLES = ["First Person", "Third Person Limited", "Third Person Omniscient", "Multiple Perspectives"]
TONES = ["Dark", "Humorous", "Inspirational", "Suspenseful", "Melancholic", "Whimsical", "Serious", "Romantic"]
LANGUAGES = ["Hungarian","English", "Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Chinese", "Russian", "Arabic"]

def render_novel_form(on_submit, button_disabled=False, button_text="Generate"):
    st.sidebar.title("Novel Generator Settings")

    # Sidebar content for model selection
    with st.sidebar:
        st.warning("🎭 AI Novel Generator: Create stories with depth and character")
        
        st.markdown("### AI Models:")
        title_agent_model = st.selectbox(
            "Title Agent Model", MODEL_LIST, index=0,
            help="Creates the novel title"
        )
        
        character_agent_model = st.selectbox(
            "Character Agent Model", MODEL_LIST, index=0,
            help="Creates detailed character profiles"
        )
        
        plot_agent_model = st.selectbox(
            "Plot Agent Model", MODEL_LIST, index=0,
            help="Creates the narrative structure and plot"
        )
        
        section_agent_model = st.selectbox(
            "Section Writer Model", MODEL_LIST, index=1,
            help="Generates the actual narrative content"
        )
        
        # Add language selection in sidebar
        st.markdown("### Language Settings:")
        language = st.selectbox(
            "Output Language", LANGUAGES, index=0,
            help="Select the language for your generated novel"
        )
        
        st.markdown("\n")
        st.image("assets/logo/powered-by-groq.svg", width=150)

    # Main form
    with st.form("novelform"):
        st.subheader("Novel Concept")
        
        if not st.session_state.get("api_key"):
            st.subheader("API Key")
            groq_input_key = st.text_input("Enter your Groq API Key (gsk_yA...):", "", type="password")
        else:
            groq_input_key = None
            
        concept_text = st.text_area(
            "Describe your novel concept",
            placeholder="E.g., 'A coming-of-age story set in a dystopian future where memories can be traded'",
            help="The core idea of your novel"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            genre = st.selectbox("Genre", GENRES)
            num_characters = st.slider("Number of Main Characters", 2, 8, 4)
        
        with col2:
            narrative_style = st.selectbox("Narrative Style", NARRATIVE_STYLES)
            tone = st.selectbox("Tone", TONES)
            
        st.subheader("Story Parameters")
        
        col3, col4 = st.columns(2)
        with col3:
            has_romance = st.checkbox("Include Romance Subplot")
            has_twist = st.checkbox("Include Plot Twist")
            
        with col4:
            complexity = st.select_slider(
                "Plot Complexity",
                options=["Simple", "Moderate", "Complex", "Intricate"]
            )
            pacing = st.select_slider(
                "Pacing",
                options=["Slow-burn", "Moderate", "Fast-paced", "Dynamic"]
            )
            
        st.subheader("Additional Instructions")
        additional_instructions = st.text_area(
            "Any specific guidelines for your novel",
            placeholder="E.g., 'Emphasize themes of betrayal and redemption', 'Set in a world inspired by medieval Japan'",
            value=""
        )
        
        st.subheader("Character Seeds (Optional)")
        character_seeds = st.text_area(
            "Provide any character ideas or starting points",
            placeholder="E.g., 'The protagonist should be a reluctant hero with a mysterious past'",
            height=150,
            value=""
        )

        st.markdown("### Story Arc")
        narrative_arc = st.selectbox(
            "Choose a narrative arc for your story",
            [
                "auto", 
                "rags_to_riches", 
                "riches_to_rags", 
                "man_in_hole", 
                "icarus", 
                "cinderella", 
                "oedipus"
            ],
            format_func=lambda x: {
                "auto": "Auto (Based on genre)",
                "rags_to_riches": "Rags to Riches (Rise)",
                "riches_to_rags": "Riches to Rags (Fall)",
                "man_in_hole": "Man in a Hole (Fall then Rise)",
                "icarus": "Icarus / Freytag's Pyramid (Rise then Fall)",
                "cinderella": "Cinderella (Rise then Fall then Rise)",
                "oedipus": "Oedipus (Fall then Rise then Fall)"
            }.get(x, x),
            help="Select a narrative arc to shape your story's emotional trajectory",
            index=0
        )

        submitted = st.form_submit_button(
            button_text,
            on_click=on_submit,
            disabled=button_disabled,
        )

    return (
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
    ) 