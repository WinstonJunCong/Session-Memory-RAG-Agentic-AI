# Future Enhancements

Planned and potential enhancements for the Session Memory Q&A Agent.

---


## 1. Streaming Responses

Stream LLM responses token-by-token instead of waiting for complete response.

```python
# Current
response = Settings.llm.complete(prompt)

# Desired
for chunk in Settings.llm.stream(prompt):
    print(chunk, end="", flush=True)
```

**Benefits**:
- Immediate feedback during long responses
- Better perceived latency
- More engaging user experience

---

## 2. Multi-Format Document Support

Expand beyond `.txt` and `.md` files.
Examples like : PDF, Word, PowerPoint, Excel, Video, Images


---

## 3. Web Search Integration Tool

Fallback to web search when knowledge base is empty.

**Notes**: Will have to implement tool calling feature as well


---

## 4. Named Collection Folders

Human-readable folder names instead of UUIDs. Can implement in a form of Sidecar Json

**Example**:

```python
# data/chroma_db/.collection_map.json
{
  "qna_docs": "52226ffb-cb80-4990-bf7c-f87ad5da4bb6",
  "conversation_memory": "9d8e7f6a-bcde-4f12-8901-2345678abcde"
}
```

**Benefits**:
- Human-readable inspection
- Zero risk to ChromaDB internals
- Cross-platform compatibility


---
## 5. Incremental Index Updates

Update index without full rebuild.

```python
# Current
build_index()  # Deletes all, rebuilds from scratch

# Desired
update_index(new_documents)  # Adds only new documents
```

**Benefits**:
- Faster for small updates
- Preserves history
- Less API cost
---

### 6. User Profiles / Multi-User

Support multiple users with separate memories.

```python
# Current
memory = get_memory()  # Global singleton

# Desired
memory = get_memory(user_id="alice")
```
---

## 7. Conversation Summarization / Fact Extraction

Summarize old messages to save tokens.

Currently memories are saved in a form of in memory history and vector db. It will be too much to brute force feed in a large amount of recent message to prompt. 

Constantly extract facts and save it in a persistant DB or do a summarization when if reaches a memory threshold.

**Benefits**:
- Handle longer conversations
- Save context tokens
- Focus on important details

---

## 8. Redis In-Memory Storage

Replace in-memory list with Redis for conversation history.

Benefits:
- Fast reads/writes 
- Cross-session persistence (survives restarts)
- Structured data types (lists, hashes)

---

## 9. PII Detection & Redaction

Detect and redact PII at ingestion time (names, IC numbers, emails).

Benefits:
- Protect sensitive user data
- Clean embeddings (no PII leakage)

---

## 10. User Review & Feedback Loop

Users rate generation quality. Collect feedback for continuous improvement.

Store ratings in ChromaDB, periodic batch review for retraining

Benefits:
- Continuous learning from user corrections/reviews
- Improve answer quality over time

---

## 11. Knowledge Base Maintenance

Scheduled cron job to refresh/rebuild index, remove stale content.

Benefits:
- Keep knowledge base current
- Remove outdated information
- Automated freshness checks


---

## 12. Observation & Logging

Structured logging and monitoring for debugging and metrics.

Benefits:
- Track query patterns and performance
- Debug issues faster
- Measure retrieval accuracy


---
