# slide_deck_splitting.py
from langchain_core.documents import Document

# 1. Define a sample text with slide separators
# The text is structured such that it contains multiple slides separated by "---"
text = """Title Slide
---
Overview of Project
---
Results and Discussion"""

# 2. Split the text into chunks using the custom separator "---"
# Each chunk represents a slide in the presentation.
slides = text.split('---')
chunks = [Document(page_content=slide.strip(), metadata={"slide": i + 1}) for i, slide in enumerate(slides)]

# 3. Print the resulting Document objects
# Each Document represents a slide with its content and metadata indicating the slide number.
for doc in chunks:
    print(f"[Slide {doc.metadata['slide']}] {doc.page_content}")
