import logging
import os
from dotenv import load_dotenv
import streamlit as st
from src.helpers import (
    is_api_key_valid,
    get_available_models,
    get_default_config_value,
)

# Load environment variables from a .env file
load_dotenv()

GENERAL_ERROR_MESSAGE = "An unexpected error occurred. If you are a developer and run the app locally, you can view the logs to see details about the error."

def is_api_key_set() -> bool:
    """Checks whether the OpenAI API key is set in streamlit's session state or as environment variable."""
    if os.getenv("OPENAI_API_KEY") or "openai_api_key" in st.session_state:
        return True
    return False

def is_temperature_and_top_p_altered() -> bool:
    """Check if both temperature and top_p settings are altered from their default values."""
    return (
        st.session_state.temperature != get_default_config_value("temperature")
        and st.session_state.top_p != get_default_config_value("top_p")
    )


def display_model_settings_sidebar():
    """Display the sidebar for adjusting model settings."""
    if "model" not in st.session_state:
        st.session_state.model = get_default_config_value("default_model.gpt")

    with st.sidebar:
        st.header("Model settings")
        model = st.selectbox(
            label="Select a large language model",
            options=get_available_models(
                model_type="gpts", api_key=st.session_state.openai_api_key
            ),
            key="model",
            help=get_default_config_value("help_texts.model"),
        )
        st.slider(
            label="Adjust temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="temperature",
            value=get_default_config_value("temperature"),
            help=get_default_config_value("help_texts.temperature"),
        )
        st.slider(
            label="Adjust Top P",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="top_p",
            value=get_default_config_value("top_p"),
            help=get_default_config_value("help_texts.top_p"),
        )
        if is_temperature_and_top_p_altered():
            st.warning(
                "OpenAI generally recommends altering temperature or top_p but not both. See their [API reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature)"
            )
        if model != get_default_config_value("default_model.gpt"):
            st.warning(
                """:warning: More advanced models (like gpt-4 and gpt-4o) have better reasoning capabilities and larger context windows. However, they likely won't make
                a big difference for short videos and simple tasks, like plain summarization. Also, beware of the higher costs of other [flagship models](https://platform.openai.com/docs/models/flagship-models)."""
            )


def display_link_to_repo(view: str = "main"):
    """Display a link to the source code repository in the sidebar."""
    st.sidebar.write(
        f"[View the source code]({get_default_config_value(f'github_repo_links.{view}')} )"
    )


def display_video_url_input(label: str = "Enter URL of the YouTube video:", disabled=False):
    """Displays an input field for the URL of the YouTube video."""
    return st.text_input(
        label=label,
        key="url_input",
        disabled=disabled,
        help=get_default_config_value("help_texts.youtube_url"),
    )


def display_yt_video_container(video_title: str, channel: str, url: str):
    """Display the YouTube video container with the video title and channel."""
    st.subheader(
        f"'{video_title}' from {channel}.",
        divider="gray",
    )
    st.video(url)


def display_nav_menu():
    """Displays links to pages in the sidebar."""
    st.sidebar.page_link(page="pages/summary.py", label="Summary")
    st.sidebar.page_link(page="pages/chat.py", label="Chat")


def is_api_key_set() -> bool:
    """Checks whether the OpenAI API key is set in session state or as an environment variable."""
    return bool(os.getenv("OPENAI_API_KEY") or "openai_api_key" in st.session_state)


def display_api_key_warning():
    """Checks whether an API key is provided and displays a warning if not."""
    if not is_api_key_set():
        st.warning(
            """:warning: It seems you haven't provided an API key yet. Make sure to do so by providing it in the settings (sidebar) 
            or as an environment variable according to the [instructions](https://github.com/sudoleg/ytai?tab=readme-ov-file#installation--usage).
            Also, make sure that you have **active credit grants** and that they are not expired! You can check it [here](https://platform.openai.com/usage),
            it should be on the right side."""
        )
    elif "openai_api_key" in st.session_state and not is_api_key_valid(st.session_state.openai_api_key):
        st.warning("API key seems to be invalid.")


def set_api_key_in_session_state():
    """Set the API key in session state from environment variable or input field."""
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.sidebar.text_input(
            "Enter your OpenAI API key",
            key="openai_api_key",
            type="password",
        )
    else:
        st.session_state.openai_api_key = OPENAI_API_KEY


def main():
    """Main function to configure and run the Streamlit app."""
    st.set_page_config(
        page_title="YouTube-Echo",
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
        st.error("The API Key you have set is not valid. Please check it and try again.")
    else:
        st.success("API Key is set and valid!")

if __name__ == "__main__":
    # Configure logging settings
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO
    )

    # Initialize session state with API key if set in environment variables
    set_api_key_in_session_state()

    # Display warnings or the main UI based on the API key status
    display_api_key_warning()

    # Display the model settings sidebar
    display_model_settings_sidebar()
