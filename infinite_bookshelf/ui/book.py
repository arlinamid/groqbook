"""
Class to store and display structured book
"""

import streamlit as st


class Book:
    def __init__(self, book_title, structure):
        self.book_title = book_title
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {
            title: st.empty() for title in self.flatten_structure(structure)
        }
        st.markdown(f"# {self.book_title}")
        st.markdown("## Generating the following:")
        toc_columns = st.columns(4)
        self.display_toc(self.structure, toc_columns)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure

        for title, content in structure.items():
            if self.contents[title].strip():  # Only display title if there is content
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        """
        Returns the markdown styled pure string with the contents.
        """
        if structure is None:
            structure = self.structure

        if level == 1:
            markdown_content = f"# {self.book_title}\n\n"

        else:
            markdown_content = ""

        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content

    def add_section_content(self, title, content):
        """
        Add content to a specific section.
        If the section doesn't exist, log a warning and skip it.
        """
        # Check if this is a metadata field that should be ignored
        metadata_fields = [
            "narrative_arc", "emotional_tone", "characters_involved", 
            "plot_structure", "narrative_advancement", "EXPOSITION", 
            "INCITING INCIDENT", "RISING ACTION", "MIDPOINT", 
            "COMPLICATIONS", "CLIMAX", "RESOLUTION"
        ]
        
        # Skip metadata fields that aren't actual sections
        if any(title.lower() == field.lower() for field in metadata_fields):
            print(f"Skipping metadata field: {title}")
            return
        
        # Try to find the section in the book structure
        found = False
        for chapter in self.structure:
            if chapter == title:
                self.contents[title] = content
                found = True
                break
            elif isinstance(self.structure[chapter], dict):
                for section in self.structure[chapter]:
                    if section == title:
                        if chapter not in self.contents:
                            self.contents[chapter] = {}
                        self.contents[chapter][section] = content
                        found = True
                        break
        
        if not found:
            print(f"Warning: Section '{title}' not found in book structure. Skipping content addition.")
