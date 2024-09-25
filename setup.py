from setuptools import setup, find_packages

setup(
    name='YouTube-Echo',
    version='0.1.0',
    description='A FastAPI application for summarizing YouTube videos.',
    author='Mohit Kumar',
    author_email='mohitpanghal12345@gmail.com',
    packages=find_packages(),
    install_requires=[
        'youtube-transcript-api',
        'langchain-community',
        'langchain-openai',
        'langchain-text-splitters',
        'streamlit',
        'watchdog',
        'pytest',
        'peewee',
        'python-dotenv',
        'chromadb',
        'langchain-chroma',
        'randomname',
        'tiktoken',
        'openai-whisper',
        'pytubefix',
    ],
    entry_points={
        'console_scripts': [
            'youtube-echo=app:app',  # Adjust if the entry point should be different
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)