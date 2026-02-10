# spilit_based_on_speaker.py
# This code demonstrates how to split a transcript based on speakers
# and create Document objects for each speaker's message.
from langchain.schema import Document

# 1. Define a sample transcript with speaker identifiers
# The transcript is structured such that each line starts with a speaker's name followed by a colon
transcript = """Alice: Let's begin the meeting.
Bob: Sure, the agenda today is RAG implementation.
Alice: Great, I have some updates."""

# 2. Split the transcript into chunks based on speaker identifiers
# Each chunk represents a message from a specific speaker.
chunks = []
for line in transcript.splitlines():
    if ':' in line:
        speaker, message = line.split(':', 1)
        chunks.append(Document(page_content=message.strip(), metadata={"speaker": speaker.strip()}))
        
# 3. Print the resulting Document objects
# Each Document represents a message from a speaker with its content and metadata indicating the speaker's name
for doc in chunks:
    print(f"[{doc.metadata['speaker']}] {doc.page_content}")
