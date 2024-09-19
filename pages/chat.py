import logging
from datetime import datetime as dt

import chromadb
import randomname
import streamlit as st
from chromadb import Collection
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from modules.helpers import (
    get_available_models,
    get_default_config_value,
    is_api_key_set,
    is_api_key_valid,
    is_environment_prod,
    num_tokens_from_string,
    save_response_as_file,
)
from modules.persistance import SQL_DB, Transcript, Video, delete_video
from modules.rag import (
    CHUNK_SIZE_TO_K_MAPPING,
    embed_excerpts,
    find_relevant_documents,
    generate_response,
    split_text_recursively,
)
from modules.transcription import download_mp3, generate_transcript
from modules.ui import (
    GENERAL_ERROR_MESSAGE,
    display_api_key_warning,
    display_link_to_repo,
    display_model_settings_sidebar,
    display_nav_menu,
    display_video_url_input,
    display_yt_video_container,
    set_api_key_in_session_state,
)
from modules.youtube import (
    InvalidUrlException,
    NoTranscriptReceivedException,
    extract_youtube_video_id,
    fetch_youtube_transcript,
    get_video_metadata,
)

CHUNK_SIZE_FOR_UNPROCESSED_TRANSCRIPT = 512

st.set_page_config("Chat", layout="wide", initial_sidebar_state="auto")
display_api_key_warning()

# --- part of the sidebar which doesn't require an api key ---
display_nav_menu()
set_api_key_in_session_state()
display_link_to_repo("chat")
# --- end ---

# --- SQLite stuff ---
SQL_DB.connect(reuse_if_open=True)
# create tables if they don't already exist
SQL_DB.create_tables([Video, Transcript], safe=True)
# --- end ---

# --- Chroma ---
chroma_connection_established = False
chroma_settings = Settings(allow_reset=True, anonymized_telemetry=False)
collection: None | Collection = None
try:
    chroma_client = chromadb.HttpClient(
        host="chromadb" if is_environment_prod() else "localhost",
        settings=chroma_settings,
    )
except Exception as e:
    logging.error(e)
    st.warning(
        "Connection to ChromaDB could not be established! You need to have a ChromaDB instance up and running locally on port 8000!"
    )
else:
    chroma_connection_established = True
# --- end ---


def is_video_selected():
    return True if selected_video_title else False


@st.dialog("Action successful")
def refresh_page(message: str):
    st.info(message)
    refresh_page_button = st.button("Refresh page")
    if refresh_page_button:
        st.session_state.url_input = ""
        st.rerun()


if (
    is_api_key_set()
    and is_api_key_valid(st.session_state.openai_api_key)
    and chroma_connection_established
):
    # --- rest of the sidebar, which requires a set api key ---
    display_model_settings_sidebar()
    st.sidebar.info(
        "Choose **text-embedding-3-large** if your video is **not** in English!"
    )
    selected_embeddings_model = st.sidebar.selectbox(
        label="Select an embedding model",
        options=get_available_models(
            api_key=st.session_state.openai_api_key, model_type="embeddings"
        ),
        key="embeddings_model",
        help=get_default_config_value(key_path="help_texts.embeddings"),
    )
    # --- end ---

    # --- initialize OpenAI models ---
    openai_chat_model = ChatOpenAI(
        api_key=st.session_state.openai_api_key,
        temperature=st.session_state.temperature,
        model=st.session_state.model,
        top_p=st.session_state.top_p,
        max_tokens=2048,
    )
    openai_embedding_model = OpenAIEmbeddings(
        api_key=st.session_state.openai_api_key,
        model=st.session_state.embeddings_model,
    )
    # --- end ---

    # fetch saved videos from SQLite
    saved_videos = Video.select()

    # create columns
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        selected_video_title = st.selectbox(
            label="Select from already processed videos",
            placeholder="Choose a video",
            options=[video.title for video in saved_videos],
            index=None,
            key="selected_video",
            help=get_default_config_value("help_texts.selected_video"),
        )
        url_input = display_video_url_input(
            label="Or enter the URL of a new video:", disabled=is_video_selected()
        )

        saved_video = None
        if is_video_selected():
            saved_video = Video.get(Video.title == selected_video_title)

        process_button = st.button(
            label="Process",
            key="process_button",
            help="This will process the transcript to enable Q&A on the contents.",
            disabled=is_video_selected(),
        )

        if saved_video:
            display_yt_video_container(
                video_title=saved_video.title,
                channel=saved_video.channel,
                url=saved_video.link,
            )
            delete_video_button = st.button(
                label="Delete",
                key="delete_video_button",
                help="Deletes selected video. You won't be able to Q&A this video, unless you process it again!",
            )
            collection = chroma_client.get_collection(
                name=saved_video.chroma_collection_name(),
            )
            if delete_video_button:
                try:
                    chroma_client.delete_collection(
                        name=saved_video.chroma_collection_name(),
                    )
                    delete_video(
                        video_title=selected_video_title,
                    )
                except Exception as e:
                    logging.error("An unexpected error occurred %s", str(e))
                    st.error(GENERAL_ERROR_MESSAGE)
                finally:
                    collection = None
                    refresh_page(
                        message=f"The video '{selected_video_title}' was deleted!"
                    )

        with st.expander("Advanced options"):
            chunk_size = st.radio(
                label="Chunk size",
                key="chunk_size",
                options=[128, 256, 512, 1024],
                index=2,
                help=get_default_config_value("help_texts.chunk_size"),
                disabled=is_video_selected(),
            )
            transcription_checkbox = st.checkbox(
                label="Enable advanced transcription",
                key="preprocessing_checkbox",
                help=get_default_config_value("help_texts.preprocess_checkbox"),
                disabled=is_video_selected(),
            )

        if process_button:
            with st.spinner(
                text="Preparing your video :gear: This can take a little, hang on..."
            ):
                try:
                    video_metadata = get_video_metadata(url_input)
                    display_yt_video_container(
                        video_title=video_metadata["name"],
                        channel=video_metadata["channel"],
                        url=url_input,
                    )

                    # 1. fetch transcript from youtube
                    saved_video = Video.create(
                        yt_video_id=extract_youtube_video_id(url_input),
                        link=url_input,
                        title=video_metadata["name"],
                        channel=video_metadata["channel"],
                        saved_on=dt.now(),
                    )
                    original_transcript = fetch_youtube_transcript(url_input)

                    saved_transcript = Transcript.create(
                        video=saved_video,
                        original_token_num=num_tokens_from_string(
                            string=original_transcript,
                            model=openai_chat_model.model_name,
                        ),
                    )

                    collection = chroma_client.get_or_create_collection(
                        name=randomname.get_name(),
                        metadata={
                            "yt_video_title": saved_video.title,
                            "chunk_size": chunk_size,
                            "embeddings_model": selected_embeddings_model,
                        },
                    )

                    # 2. create excerpts. Either
                    #   - from original transcript
                    #   - or from whisper transcription if transcription checkbox is checked
                    if transcription_checkbox:
                        download = download_mp3(
                            video_id=saved_video.yt_video_id,
                            download_folder_path="data/audio",
                        )
                        whisper_transcript = generate_transcript(file_path=download)
                        save_response_as_file(
                            dir_name="data/transcripts",
                            filename=saved_video.title,
                            file_content=whisper_transcript,
                        )
                        transcript_excerpts = split_text_recursively(
                            transcript_text=whisper_transcript,
                            chunk_size=chunk_size,
                            len_func="tokens",
                        )
                    else:
                        transcript_excerpts = split_text_recursively(
                            transcript_text=original_transcript,
                            chunk_size=chunk_size,
                            len_func="tokens",
                        )

                    # 3. embed/index transcript excerpts
                    Transcript.update(
                        {
                            Transcript.preprocessed: transcription_checkbox,
                            Transcript.chunk_size: chunk_size,
                            Transcript.chroma_collection_id: collection.id,
                            Transcript.chroma_collection_name: collection.name,
                        }
                    ).where(Transcript.video == saved_video).execute()
                    embed_excerpts(
                        collection=collection,
                        excerpts=transcript_excerpts,
                        embeddings=openai_embedding_model,
                    )
                except InvalidUrlException as e:
                    st.error(e.message)
                    e.log_error()
                except NoTranscriptReceivedException as e:
                    st.error(e.message)
                    e.log_error()
                except Exception as e:
                    logging.error(
                        "An unexpected error occurred: %s", str(e), exc_info=True
                    )
                    st.error(GENERAL_ERROR_MESSAGE)
                else:
                    refresh_page(
                        message="The video has been processed! Please refresh the page and choose it in the select-box above."
                    )

    with col2:
        if collection and collection.count() > 0:
            # the users input has to be embedded using the same embeddings model as was used for creating
            # the embeddings for the transcript excerpts. Here we ensure that the embedding function passed
            # as argument to the vector store is the same as was used for the embeddings
            collection_embeddings_model = collection.metadata.get("embeddings_model")
            if collection_embeddings_model != selected_embeddings_model:
                openai_embedding_model.model = collection_embeddings_model

            # init vector store
            chroma_db = Chroma(
                client=chroma_client,
                collection_name=collection.name,
                embedding_function=openai_embedding_model,
            )

            prompt = st.chat_input(
                placeholder="Ask a question or provide a topic covered in the video",
                key="user_prompt",
            )

            if prompt:
                with st.spinner("Generating answer..."):
                    try:
                        relevant_docs = find_relevant_documents(
                            query=prompt,
                            db=chroma_db,
                            k=CHUNK_SIZE_TO_K_MAPPING.get(
                                collection.metadata["chunk_size"]
                            ),
                        )
                        response = generate_response(
                            question=prompt,
                            llm=openai_chat_model,
                            relevant_docs=relevant_docs,
                        )
                    except Exception as e:
                        logging.error(
                            "An unexpected error occurred: %s", str(e), exc_info=True
                        )
                        st.error(GENERAL_ERROR_MESSAGE)
                    else:
                        st.write(response)
                        with st.expander(
                            label="Show chunks retrieved from index and provided to the model as context"
                        ):
                            for d in relevant_docs:
                                st.write(d.page_content)
                                st.divider()
