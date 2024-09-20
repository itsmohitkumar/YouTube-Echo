import logging
import streamlit as st
from modules.helpers import is_api_key_set, is_api_key_valid
from modules.ui import display_link_to_repo, display_nav_menu

def main():
    """Main function to configure and run the Streamlit app."""
    st.set_page_config(
        page_title="YouTube AI",
        layout="wide",
        initial_sidebar_state="auto"
    )
    # Display sidebar with page links
    display_nav_menu()
    display_link_to_repo()

    # Check if API key is set and display appropriate message
    if not is_api_key_set():
        st.info(
            """It looks like you haven't set the API Key as an environment variable. 
            Don't worry, you can set it in the sidebar when you navigate to any of the pages :)"""
        )
    elif not is_api_key_valid():
        st.error(
            "The API Key you have set is not valid. Please check it and try again."
        )
    else:
        st.success("API Key is set and valid!")

if __name__ == "__main__":
    # Configure logging settings
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO
    )
    main()
