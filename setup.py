from setuptools import setup, find_packages

setup(
    name='YouTube-Echo',
    version='0.1.0',
    description='A FastAPI application for summarizing YouTube videos.',
    author='Mohit Kumar',
    author_email='mohitpanghal12345@gmail.com',
    packages=find_packages(),
    install_requires=[
        'youtube-transcript-api==0.6.2',
        'langchain-community==0.3.0',
        'langchain-openai==0.2.0',
        'langchain-text-splitters==0.3.0',
        'streamlit==1.38.0',
        'watchdog==2.1.5',
        'pytest==8.3.3',
        'peewee==3.17.6',
        'python-dotenv==1.0.1',
        'chromadb==0.5.7',
        'langchain-chroma==0.1.4',
        'randomname==0.2.1',
        'tiktoken==0.7.0',
        'openai-whisper==20231117',
        'pytubefix==6.17.0',
        'flask==3.0.3',
        'gunicorn==20.1.0',
        'torch==2.3.0',
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
    python_requires='3.10.4',
)
