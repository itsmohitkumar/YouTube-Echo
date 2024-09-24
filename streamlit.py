import logging
import os
from typing import List, Literal
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_openai import ChatOpenAI
from src.app.helpers import (
    TranscriptException,
    TranscriptProcessor,
)
from src.app.helpers import (
    ConfigManager,
    FileManager,
)
from src.app.youtube import (
    InvalidUrlException,
    NoTranscriptReceivedException,
    YouTubeTranscriptManager,
)

# Load environment variables from a .env file
load_dotenv()

GENERAL_ERROR_MESSAGE = (
    "An unexpected error occurred. If you are a developer and run the app locally, "
    "you can view the logs to see details about the error."
)

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO
    )

def get_available_models(model_type: Literal["gpts", "embeddings"], api_key: str = "") -> List[str]:
    """
    Retrieve a filtered list of available model IDs from OpenAI's API or environment variables,
    based on the specified model type.
    """
    openai.api_key = api_key
    selectable_model_ids = list(ConfigManager.get_default_config_value(f"available_models.{model_type}"))

    available_model_ids = os.getenv("AVAILABLE_MODEL_IDS")
    if available_model_ids:
        return list(filter(lambda m: m in available_model_ids.split(","), selectable_model_ids))

    try:
        available_model_ids = [model.id for model in openai.models.list()]
        os.environ["AVAILABLE_MODEL_IDS"] = ",".join(available_model_ids)
        return list(filter(lambda m: m in available_model_ids, selectable_model_ids))
    except openai.AuthenticationError as e:
        logging.error("An authentication error occurred: %s", str(e))
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
    
    return []

def is_api_key_valid(api_key: str) -> bool:
    """Checks the validity of an OpenAI API key."""
    openai.api_key = api_key
    try:
        openai.models.list()
        logging.info("API key validation successful")
        os.environ["OPENAI_API_KEY_VALID"] = "yes"
        return True
    except openai.AuthenticationError as e:
        logging.error("An authentication error occurred: %s", str(e))
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
    
    return False

def is_api_key_set() -> bool:
    """Checks if the OpenAI API key is set in Streamlit's session state or as an environment variable."""
    return bool(os.getenv("OPENAI_API_KEY") or "openai_api_key" in st.session_state)

def is_temperature_and_top_p_altered() -> bool:
    """Check if both temperature and top_p settings are altered from their default values."""
    return (
        st.session_state.temperature != ConfigManager.get_default_config_value("temperature") and
        st.session_state.top_p != ConfigManager.get_default_config_value("top_p")
    )

def display_model_settings_sidebar():
    """Display the sidebar for adjusting model settings."""
    if "model" not in st.session_state:
        st.session_state.model = ConfigManager.get_default_config_value("default_model.gpt")

    with st.sidebar:
        st.header("Model settings")
        model = st.selectbox(
            label="Select a large language model",
            options=get_available_models(model_type="gpts", api_key=st.session_state.openai_api_key),
            key="model",
            help=ConfigManager.get_default_config_value("help_texts.model"),
        )
        st.slider(
            label="Adjust temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="temperature",
            value=ConfigManager.get_default_config_value("temperature"),
            help=ConfigManager.get_default_config_value("help_texts.temperature"),
        )
        st.slider(
            label="Adjust Top P",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="top_p",
            value=ConfigManager.get_default_config_value("top_p"),
            help=ConfigManager.get_default_config_value("help_texts.top_p"),
        )
        if is_temperature_and_top_p_altered():
            st.warning(
                "OpenAI generally recommends altering temperature or top_p but not both. "
                "See their [API reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature)"
            )
        if model != ConfigManager.get_default_config_value("default_model.gpt"):
            st.warning(
                "More advanced models (like gpt-4 and gpt-4o) have better reasoning capabilities and larger context windows. "
                "However, they likely won't make a big difference for short videos and simple tasks, like plain summarization. "
                "Also, beware of the higher costs of other [flagship models](https://platform.openai.com/docs/models/flagship-models)."
            )

def display_link_to_repo(view: str = "main"):
    """Display a link to the source code repository in the sidebar."""
    st.sidebar.write(
        f"[📖 View the source code]({ConfigManager.get_default_config_value(f'github_repo_links.{view}')})"
    )

def display_video_url_input(label: str = "Enter URL of the YouTube video:", disabled=False):
    """Displays an input field for the URL of the YouTube video."""
    return st.text_input(
        label=label,
        key="url_input",
        disabled=disabled,
        help=ConfigManager.get_default_config_value("help_texts.youtube_url"),
    )

def display_api_key_warning():
    """Checks whether an API key is provided and displays a warning if not."""
    if not is_api_key_set():
        st.warning(
            "It seems you haven't provided an API key yet. Make sure to do so by providing it in the settings (sidebar) "
            "or as an environment variable according to the [instructions](https://github.com/sudoleg/ytai?tab=readme-ov-file#installation--usage). "
            "Also, make sure that you have **active credit grants** and that they are not expired! You can check it [here](https://platform.openai.com/usage)."
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
    setup_logging()
    set_api_key_in_session_state()
    display_api_key_warning()

    st.set_page_config(page_title="YouTube-Echo", layout="wide", initial_sidebar_state="auto")

    # Sidebar content
    if not is_api_key_set():
        st.info(
            "It seems you haven't set your API Key as an environment variable. "
            "You can enter it in the sidebar while navigating through the pages."
        )
    elif not is_api_key_valid(st.session_state.get("openai_api_key", "")):
        st.error("The API Key you've entered is invalid. Please verify it and try again.")
    else:
        st.success("Your API Key is valid and set!")

    if is_api_key_set() and is_api_key_valid(st.session_state.openai_api_key):
        display_model_settings_sidebar()
        st.sidebar.checkbox(
            label="Save responses",
            value=False,
            help=ConfigManager.get_default_config_value("help_texts.saving_responses"),
            key="save_responses",
        )
        
        col1, col2 = st.columns([0.4, 0.6], gap="large")

        with col1:
            with st.container():
                st.markdown("<h4 style='text-align: center;'>Enter the YouTube video URL:</h4>", unsafe_allow_html=True)
                url_input = st.text_input(
                    label="",
                    key="url_input",
                    help=ConfigManager.get_default_config_value("help_texts.youtube_url"),
                    placeholder="e.g., https://www.youtube.com/watch?v=example"
                )

                st.markdown("<h4 style='text-align: center;'>Optionally, enter a custom prompt:</h4>", unsafe_allow_html=True)
                custom_prompt = st.text_area(
                    "",
                    key="custom_prompt_input",
                    help=ConfigManager.get_default_config_value("help_texts.custom_prompt"),
                )
                
                summarize_button = st.button("Summarize", key="summarize_button", help="Click here to summarize the video.")

        with col2:
            if url_input:
                try:
                    vid_metadata = YouTubeTranscriptManager.get_video_metadata(url_input)
                    if vid_metadata:
                        st.subheader(
                            f"Video Title: '{vid_metadata['name']}' from Channel: {vid_metadata['channel']}.",
                            divider="gray",
                        )
                    st.video(url_input)
                except InvalidUrlException as e:
                    st.error(e.message)
                    e.log_error()
                except Exception as e:
                    logging.error("An unexpected error occurred: %s", str(e))
                    st.error(GENERAL_ERROR_MESSAGE)

            if summarize_button and url_input:
                try:
                    transcript = YouTubeTranscriptManager.fetch_youtube_transcript(url_input)
                    cb = OpenAICallbackHandler()
                    llm = ChatOpenAI(
                        api_key=st.session_state.openai_api_key,
                        temperature=st.session_state.temperature,
                        model=st.session_state.model,
                        top_p=st.session_state.top_p,
                        callbacks=[cb],
                        max_tokens=2048,
                    )
                    with st.spinner("Summarizing video... Please wait..."):
                        transcript_processor = TranscriptProcessor(llm)
                        resp = transcript_processor.get_transcript_summary(transcript, custom_prompt=custom_prompt) if custom_prompt else transcript_processor.get_transcript_summary(transcript)
                    st.markdown(resp)
                    st.caption(f"Estimated cost for this request: {cb.total_cost:.4f}$")
                    if st.session_state.save_responses:
                        FileManager.save_response_as_file(
                            dir_name=f"./responses/{vid_metadata['channel']}",
                            filename=f"{vid_metadata['name']}",
                            file_content=resp,
                            content_type="markdown",
                        )
                except (InvalidUrlException, NoTranscriptReceivedException, TranscriptException) as e:
                    st.error(e.message)
                    e.log_error()
                except Exception as e:
                    logging.error("An unexpected error occurred: %s", str(e), exc_info=True)
                    st.error(GENERAL_ERROR_MESSAGE)

    # Link to the repository at the bottom of the sidebar
    display_link_to_repo()

if __name__ == "__main__":
    main()
