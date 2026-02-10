# json_line_chunk.py
# It demonstrates how to split a JSON Lines formatted string into Document objects.
import json
from langchain.schema import Document

# 1.Sample JSON file
json_lines = """
{"id": 1, "text": "First entry"}
{"id": 2, "text": "Second entry"}
"""

# 2. Split the JSON Lines string into individual lines,
# parse each line as a JSON object, and create Document objects.
chunks = []
for line in json_lines.strip().splitlines():
    item = json.loads(line)
    chunks.append(Document(page_content=item["text"], metadata={"id": item["id"]}))

# 3. Print the resulting Document objects
# Each Document represents a JSON object with its content and metadata.
for doc in chunks:
    print(f"[ID {doc.metadata['id']}] {doc.page_content}")
