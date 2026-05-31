<h1>Job Market Intelligence & Portfolio Generator (LLM + Vector Search)</h1> <h2>Description</h2>

This project is a job market intelligence system designed to help students and job seekers build portfolio projects aligned with real industry standards. It collects live job postings using SerpAPI, extracts and processes job descriptions, and stores them as embeddings in a persistent Chroma vector database.

The system uses semantic search and a local LLM (Ollama) to analyze clusters of job descriptions and identify hiring patterns. It then generates actionable portfolio project ideas that directly reflect what employers are actually looking for, helping users bridge the gap between learning and real-world job requirements.

<h2>Core Features</h2>
Fetches live job postings using SerpAPI (Google Jobs engine)
Extracts and aggregates job descriptions across multiple pages
Cleans and deduplicates job data
Generates embeddings using HuggingFace Sentence Transformers
Stores embeddings in ChromaDB for semantic search
Retrieves relevant job clusters based on user queries
Uses a local LLM (Ollama) to analyze hiring patterns
Generates portfolio projects aligned with real industry demand
Focused on helping students build job-relevant, industry-standard projects
<h2>Languages and Utilities Used</h2>
Python
LangChain
Ollama (phi3:mini)
ChromaDB
HuggingFace Embeddings
SerpAPI
dotenv
<h2>System Workflow</h2>
Input job title and location
Fetch live job listings from SerpAPI
Extract and aggregate job descriptions
Generate embeddings and store in ChromaDB
Retrieve relevant job clusters via semantic search
Pass context into LLM for analysis
Output portfolio project ideas aligned with industry expectations
<h2>File Access</h2>
Run the main script to start the pipeline
Configure API keys in .env
Vector data is stored locally in ./chroma_db
