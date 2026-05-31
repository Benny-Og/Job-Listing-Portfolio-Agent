import os
from dotenv import load_dotenv
load_dotenv()

import serpapi
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_ollama import ChatOllama

llm = ChatOllama(model="phi3:mini")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
client = serpapi.Client(api_key=os.getenv("SERPAPI_KEY"))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# get the description responses
def fetch_job_descriptions(query, location, max_pages=3):
    next_page_token = None
    pages = 0

    all_descriptions = []

    while pages < max_pages:

        params = {
            "engine": "google_jobs",
            "q": query,
            "location": location,
        }

        if next_page_token:
            params["next_page_token"] = next_page_token

        results = client.search(params)

        jobs = results.get("jobs_results", [])
        if not jobs:
            break

        for job in jobs:
            print(job.get("title"))
            print(job.get("company_name"))
            print(job.get("location"))
            print("-" * 40)

            desc = job.get("description")
            if desc:
                all_descriptions.append(desc)

        next_page_token = results.get("serpapi_pagination", {}).get("next_page_token")

        if not next_page_token:
            break

        pages += 1

    return all_descriptions


## append list of descriptions to vectorize
def vectorize_descriptions(descriptions):
    return embedding_model.embed_documents(descriptions)


#store vectors in chromadb
def store_in_chroma(descriptions, embeddings):

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="job_portfolio"
    )

    ids = [str(i) for i in range(len(descriptions))]

    collection.add(
        ids=ids,
        documents=descriptions,
        embeddings=embeddings
    )

    return collection


#retrieve collections
def retrieve_jobs(collection, query, k=5):
    return collection.query(
        query_embeddings=[embedding_model.embed_query(query)],
        n_results=k
    )


#generate llm response
def analyze_jobs(collection, query):

    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]

    context = "\n\n---\n\n".join(docs)

    prompt = f"""
You are a senior hiring manager and technical lead analyzing real job postings.

Below are multiple job descriptions from active listings:

{context}

---

TASK:

Based on ALL job descriptions collectively, design exactly TWO portfolio projects that would make a candidate competitive for these roles.

Each project must reflect real hiring demand found in the descriptions.

Do NOT summarize individual jobs. Instead, infer what the job market is collectively asking for and translate that into project ideas.

---

REQUIREMENTS:

- Provide exactly 2 distinct portfolio projects
- Each project must be realistic and buildable
- Each project should target different angles of the job market (e.g., one data/ML heavy, one systems/engineering heavy if applicable)
- Avoid generic ideas (no "build a dashboard" alone, no vague "analyze data")
- Make projects feel like real-world systems used in industry
- Be specific about what the system does end-to-end

---

OUTPUT FORMAT:

For each project include:

Project 1:
- Project Name
- What the project is (clear explanation of the system and what it does)
- Why this matches the job descriptions
- Skills learned (bullet list)

Project 2:
- Project Name
- What the project is (clear explanation of the system and what it does)
- Why this matches the job descriptions
- Skills learned (bullet list)

---

STYLE:

- Write in a natural, recruiter-level explanation style
- Be specific and technical, but easy to read
- Focus on real-world applicability
- Emphasize skills employers are repeatedly looking for
- Do NOT output JSON or code formatting
"""

    response = llm.invoke(prompt)

    return response.content


# run
if __name__ == "__main__":

    query = input("Job title: ")
    location = input("State: ")

    descriptions = fetch_job_descriptions(query, location)
    print(f"\nCollected {len(descriptions)} descriptions")

    vectors = vectorize_descriptions(descriptions)
    print(f"Generated {len(vectors)} vectors")

    collection = store_in_chroma(descriptions, vectors)
    print("\nStored in ChromaDB")

    user_query = input("\nAsk about these jobs (or press enter for default analysis): ")
    if not user_query.strip():
        user_query = query

    result = analyze_jobs(collection, user_query)
    print("\n===== AI ANALYSIS =====\n")
    print(result)