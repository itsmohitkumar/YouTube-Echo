import os
import ssl
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
import scrapetube

# Bypass SSL certificate verification (development only)
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class YouTubeVideo:
    def __init__(self, video_id, video_title, video_url, channel_name, duration, publish_date):
        self.video_id = video_id
        self.video_title = video_title
        self.video_url = video_url
        self.channel_name = video_name
        self.duration = duration
        self.publish_date = publish_date

class YouTubeTranscriptService:
    def __init__(self):
        self.transcript_loader = None

    def fetch_transcript(self, url):
        try:
            self.transcript_loader = YoutubeLoader.from_youtube_url(
                url, 
                add_video_info=True, 
                transcript_format=TranscriptFormat.CHUNKS, 
                chunk_size_seconds=30
            )
            return self.transcript_loader.load()
        except Exception as e:
            st.error(f"An error occurred while fetching the transcript: {str(e)}")
            return []

class VideoSearchService:
    def __init__(self):
        self.convert_sorting_option = {
            "Most Relevant": "relevance",
            "Upload Date": "upload_date",
            "View Count": "view_count", 
            "Rating": "rating"
        }

    def search_videos(self, search_term, video_count=1, sorting_criteria="relevance"):
        videos = scrapetube.get_search(
            query=search_term, 
            limit=video_count, 
            sort_by=self.convert_sorting_option[sorting_criteria]
        )
        return [YouTubeVideo(
            video_id=video["videoId"],
            video_title=video["title"]["runs"][0]["text"],
            video_url=f"https://www.youtube.com/watch?v={video['videoId']}",
            channel_name=video["longBylineText"]["runs"][0]["text"],
            duration=video["lengthText"]["accessibility"]["accessibilityData"]["label"],
            publish_date=video["publishedTimeText"]["simpleText"]
        ) for video in videos]

class EmbeddingService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    def embed_documents(self, documents):
        return self.embeddings.embed_documents(documents)

    def create_vectorstore(self, documents):
        return FAISS.from_documents(documents, self.embeddings)

class AIService:
    def __init__(self):
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",  # Adjust model name as needed
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def ask_question(self, messages):
        return self.llm.invoke(messages).content

class VideoChatApp:
    def __init__(self):
        self.transcript_service = YouTubeTranscriptService()
        self.search_service = VideoSearchService()
        self.embedding_service = EmbeddingService()
        self.ai_service = AIService()
        self.initialize_session_state()

    def initialize_session_state(self):
        if "current_video_url" not in st.session_state:
            st.session_state.current_video_url = None
            st.session_state.current_transcript_docs = []
            st.session_state.videos = []

    def rag_with_video_transcript(self, transcript_docs, prompt):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len
        )
        
        splitted_documents = text_splitter.split_documents(transcript_docs)
        
        if len(splitted_documents) == 0:
            st.error("No documents to process.")
            return "No documents to process.", []

        try:
            document_texts = [doc.page_content for doc in splitted_documents]
            document_embeddings = self.embedding_service.embed_documents(document_texts)

            if len(document_embeddings) == 0 or len(document_embeddings[0]) == 0:
                st.error("Document embeddings are empty or invalid.")
                return "Document embeddings are empty or invalid.", []

            vectorstore = self.embedding_service.create_vectorstore(splitted_documents)
            retriever = vectorstore.as_retriever()
            # Using `invoke` to get relevant documents
            relevant_documents = retriever.invoke(prompt)
            
            context_data = " ".join(doc.page_content for doc in relevant_documents)
            final_prompt = f"""I have the following question: {prompt}
            To answer this question, we have the following information: {context_data}.
            Please use only the information provided here to answer the question. Do not include any external information.
            """
            
            AI_Response = self.ai_service.ask_question([("system", "You are a helpful assistant."), ("human", final_prompt)])
            
            return AI_Response, relevant_documents

        except Exception as e:
            st.error(f"An error occurred during RAG processing: {str(e)}")
            return "An error occurred during RAG processing.", []

    def render_url_tab(self):
        video_url = st.text_input(label="Enter YouTube Video URL:", key="url_video_url")
        prompt = st.text_input(label="Enter your question:", key="url_prompt")
        submit_btn = st.button(label="Ask", key="url_btn")
        
        if submit_btn:
            st.video(data=video_url)
            st.divider()
            if st.session_state.current_video_url != video_url:
                with st.spinner(text="STEP 1: Preparing video transcript..."):
                    transcript_docs = self.transcript_service.fetch_transcript(url=video_url)
                    st.session_state.current_transcript_docs = transcript_docs
            st.success("Video transcript cached!")
            st.divider()
            st.session_state.current_video_url = video_url
            
            with st.spinner("STEP 2: Answering your question..."):
                AI_Response, relevant_documents = self.rag_with_video_transcript(transcript_docs=transcript_docs, prompt=prompt)
            st.info("ANSWER:")
            st.markdown(AI_Response)
            st.divider()
            
            for doc in relevant_documents:
                st.warning("REFERENCE:")
                st.caption(doc.page_content)
                st.markdown(f"Source: {doc.metadata}")
                st.divider()

    def render_search_tab(self):
        col_left, col_center, col_right = st.columns([20,1,10])

        with col_left:
            st.subheader("Video Search")
            st.divider()
            search_term = st.text_input(label="Enter search keywords:", key="search_term")
            video_count = st.slider(label="Number of results", min_value=1, max_value=5, value=5, key="search_video_count")
            sorting_options = ["Most Relevant", "Upload Date", "View Count", "Rating"]
            sorting_criteria = st.selectbox(label="Sort by", options=sorting_options)
            search_btn = st.button(label="Search Videos", key="search_button")
            st.divider()

            if search_btn:
                st.session_state.videos = self.search_service.search_videos(search_term=search_term, video_count=video_count, sorting_criteria=sorting_criteria)
                
            video_urls = [video.video_url for video in st.session_state.videos]
            video_titles = {video.video_url: video.video_title for video in st.session_state.videos}

            selected_video = st.selectbox(
                label="Select a video to chat about:",
                options=video_urls,
                format_func=lambda url: video_titles[url],
                key="search_selectbox"
            )
                
            if selected_video:
                search_prompt = st.text_input(label="Enter your question:", key="search_prompt")
                search_ask_btn = st.button(label="Ask", key="search_ask_button")

                if search_ask_btn:
                    st.caption("Selected Video")
                    st.video(data=selected_video)
                    st.divider()
                        
                    if st.session_state.current_video_url != selected_video:
                        with st.spinner("STEP 1: Preparing video transcript..."):
                            video_transcript_docs = self.transcript_service.fetch_transcript(url=selected_video)
                            st.session_state.current_transcript_docs = video_transcript_docs
                        st.success("Video transcript cached!")
                        st.divider()
                        st.session_state.current_video_url = selected_video
                
                    with st.spinner("STEP 2: Answering your question..."):
                        AI_Response, relevant_documents = self.rag_with_video_transcript(transcript_docs=st.session_state.current_transcript_docs, prompt=search_prompt)
                    st.info("ANSWER:")
                    st.markdown(AI_Response)
                    st.divider()

                    for doc in relevant_documents:
                        st.warning("REFERENCE:")
                        st.caption(doc.page_content)
                        st.markdown(f"Source: {doc.metadata}")
                        st.divider()

        with col_center:
            st.empty()

        with col_right:
            st.subheader("Related Videos")
            st.divider()

            for i, video in enumerate(st.session_state.videos):
                st.info(f"Video No: {i+1}")
                st.video(data=video.video_url)
                st.caption(f"Video Title: {video.video_title}")
                st.caption(f"Channel: {video.channel_name}")
                st.caption(f"Duration: {video.duration}")
                st.divider()

    def run(self):
        st.set_page_config(page_title="VidChat: Chat with YouTube", layout="centered")

        st.header("VidChat: Chat with YouTube!")
        st.divider()
        tab_urls, tab_search = st.tabs(["By URL", "By Search"])

        with tab_urls:
            self.render_url_tab()

        with tab_search:
            self.render_search_tab()

if __name__ == "__main__":
    app = VideoChatApp()
    app.run()
