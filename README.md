# Commerce Assistant Prototype

Minimal RAG pipeline: feed docs → chunk → store → retrieve → answer.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add your Anthropic API key
```
