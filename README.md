# Qwen Personal AI - Advanced Permanent Memory

A local personal AI assistant powered by Ollama, with permanent memory, multi-conversation support, web search, and web page analysis capabilities.

## Features

- **Permanent Memory**: Remembers important facts about you across all conversations
- **Multi-Conversation System**: Create, switch, rename, and delete separate conversation threads
- **Web Search**: Research topics on the internet and get summarized answers with sources
- **Web Page Reader**: Analyze the content of any webpage
- **Semantic Memory Search**: Find relevant past conversations using embeddings
- **Lightweight Model**: Designed to work well with small models like `qwen3.5:0.8b`

## Requirements

  - Python 3.10+
  - [Ollama](https://ollama.com/) installed and running
  - Models:
  - Chat model: `qwen3.5:0.8b` (or any other Ollama model)
  - Embedding model: `nomic-embed-text`

## Installation

## 1. Clone the repository:

    git clone https://github.com/sourcecode347/qwen-personal-ai.git
    cd qwen-personal-ai

## 2. Setup Python Virtual Environment 

    python -m venv venv
    source venv/bin/activate

## 3. Install dependencies

    pip install requests numpy beautifulsoup4 lxml duckduckgo-search

## 4. Pull the required models with Ollama

    ollama pull qwen3.5:0.8b
    ollama pull nomic-embed-text

## Usage

    python qwen3.5:0.8b.py

## Available Commands

+------------------+------------------------------------+
| Command          | Description                        |
| ---------------- | ---------------------------------- |
| /new TITLE       | Create a new conversation          |
| /chats           | List all conversations             |
| /open ID         | Switch to a conversation           |
| /rename ID TITLE | Rename a conversation              |
| /delete ID       | Delete a conversation              |
| /web TEXT        | Search the web and analyze results |
| /read URL        | Read and analyze a webpage         |
| /search TEXT     | Search past conversations          |
| /facts           | Show permanent memory              |
| /forget ID       | Delete a fact from memory          |
| /history         | Show recent messages               |
| /count           | Show statistics                    |
| /paste           | Enter multi-line mode              |
| /help            | Show help                          |
| /exit            | Exit                               |
+------------------+------------------------------------+

## Examples

YOU: What is the capital of France?

YOU: /web latest AI news

YOU: /read https://example.com

YOU: /new Project Ideas

## How It Works

Permanent Memory: Important facts you share are automatically extracted and stored in a SQLite database. These facts are injected into future conversations.

Conversations: Each conversation is isolated, but permanent facts are shared across all conversations.

Web Search: Uses DuckDuckGo to search the web, visits the top results, extracts text, and asks the model to summarize the findings.

Embeddings: Uses nomic-embed-text to find semantically similar past messages.

## Configuration

You can modify the following variables at the top of the script:

MODEL: The Ollama model used for chat
EMBED_MODEL: The embedding model
DB: Path to the SQLite database
TOP_K, HISTORY_LIMIT, FACT_LIMIT: Memory and history settings

## Database

The script creates a local SQLite database (qwen3.5:0.8b.db by default) that stores:

Messages
Permanent facts
Conversations
Settings
