🧠 Simple Knowledge Graph RAG with Neo4j
Load a FAQ file → Extract entities & relations → Store in Neo4j → Ask questions with AI

🔄 How It Works
FAQ.txt → LLM extracts entities & relations → Stored in Neo4j → Ask questions → AI answers!
⚡ Quick Setup (5 Minutes)
1. Create Neo4j Database (Free)
Go to neo4j.com/cloud/aura-free
Sign up → Create AuraDB Free instance
Save your URI, Username, Password
2. Create Project
mkdir kg-rag && cd kg-rag
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3. Install Dependencies
pip install openai neo4j python-dotenv
4. Create .env File
OPENAI_API_KEY=sk-your-key-here
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
5. Create faq.txt
Q: What is AI?
A: Artificial Intelligence is a branch of computer science created by Alan Turing in 1950. Major labs include OpenAI, Google DeepMind, and Anthropic.

Q: What is Machine Learning?
A: Machine Learning is a subset of AI that learns from data. Popular frameworks include TensorFlow by Google and PyTorch by Meta.

Q: What is Deep Learning?
A: Deep Learning is a subset of Machine Learning using neural networks. Pioneers include Geoffrey Hinton and Yann LeCun. It powers image recognition and language models.

Q: What is NLP?
A: Natural Language Processing is a field of AI for human-computer language interaction. Key models include BERT by Google and GPT by OpenAI.

Q: What is a Knowledge Graph?
A: A Knowledge Graph stores entities and their relationships. Google introduced it in 2012. Neo4j is the most popular graph database for building them.
6. Create app.py (Single File — That's It!)
import os, json
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

# ─────────────────────────────────────────────
# STEP 1: Load Document
# ─────────────────────────────────────────────
def load_faq(path="faq.txt"):
    with open(path) as f:
        text = f.read()
    # Split into chunks by Q&A pairs
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    print(f"📄 Loaded {len(chunks)} chunks")
    return chunks

# ─────────────────────────────────────────────
# STEP 2: Extract Entities & Relations (LLM)
# ─────────────────────────────────────────────
def extract(chunk):
    prompt = f"""Extract entities and relations from this text.
Return ONLY valid JSON like:
{{"entities": [{{"name": "AI", "type": "CONCEPT"}}], "relations": [{{"source": "ML", "relation": "SUBSET_OF", "target": "AI"}}]}}

Text: {chunk}"""

    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = res.choices[0].message.content.strip()
    # Clean markdown code blocks
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(text)
    except:
        return {"entities": [], "relations": []}

# ─────────────────────────────────────────────
# STEP 3: Store in Neo4j
# ─────────────────────────────────────────────
def store(entities, relations):
    with db.session() as s:
        s.run("MATCH (n) DETACH DELETE n")  # Clear old data
        for e in entities:
            s.run("MERGE (n:Entity {name: $name}) SET n.type = $type", name=e["name"], type=e["type"])
        for r in relations:
            s.run("""
                MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                MERGE (a)-[:RELATES {type: $rel}]->(b)
            """, src=r["source"], rel=r["relation"], tgt=r["target"])
    print(f"💾 Stored {len(entities)} entities, {len(relations)} relations")

# ─────────────────────────────────────────────
# STEP 4: Query (RAG)
# ─────────────────────────────────────────────
def search_graph(question):
    # Ask LLM to extract keywords
    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f'Extract keywords from this question as a JSON array: "{question}"'}],
        temperature=0
    )
    text = res.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    keywords = json.loads(text)

    # Search Neo4j for matching entities + their connections
    results = []
    with db.session() as s:
        for kw in keywords:
            records = s.run("""
                MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($kw)
                OPTIONAL MATCH (n)-[r:RELATES]->(m)
                OPTIONAL MATCH (p)-[r2:RELATES]->(n)
                RETURN n.name AS entity, n.type AS type,
                       collect(DISTINCT {rel: r.type, target: m.name}) AS out,
                       collect(DISTINCT {rel: r2.type, source: p.name}) AS inc
            """, kw=kw)
            for rec in records:
                d = rec.data()
                info = f"{d['entity']} ({d['type']})"
                for o in d['out']:
                    if o['target']: info += f"\n  → {d['entity']} --{o['rel']}--> {o['target']}"
                for i in d['inc']:
                    if i['source']: info += f"\n  ← {i['source']} --{i['rel']}--> {d['entity']}"
                results.append(info)
    return "\n\n".join(results) if results else "No info found."

def ask(question):
    context = search_graph(question)
    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based ONLY on this Knowledge Graph context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    return res.choices[0].message.content

# ─────────────────────────────────────────────
# RUN THE PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # === BUILD PHASE ===
    print("🚀 Building Knowledge Graph...\n")
    chunks = load_faq()

    all_entities, all_relations = [], []
    seen_e, seen_r = set(), set()

    for i, chunk in enumerate(chunks):
        print(f"  🧠 Extracting chunk {i+1}/{len(chunks)}...")
        data = extract(chunk)
        for e in data["entities"]:
            if e["name"] not in seen_e:
                seen_e.add(e["name"])
                all_entities.append(e)
        for r in data["relations"]:
            key = (r["source"], r["relation"], r["target"])
            if key not in seen_r:
                seen_r.add(key)
                all_relations.append(r)

    store(all_entities, all_relations)
    print("✅ Knowledge Graph ready!\n")

    # === QUERY PHASE ===
    print("💬 Ask anything! (type 'quit' to exit)\n")
    while True:
        q = input("❓ Question: ").strip()
        if q.lower() in ["quit", "exit", "q"]: break
        if q:
            print(f"\n💡 {ask(q)}\n")

    db.close()
7. Run It!
python app.py
🖼️ See Your Graph
Open Neo4j Browser and run:

MATCH (n)-[r]->(m) RETURN n, r, m
You'll see something like:

(AI) ←──SUBSET_OF── (Machine Learning) ←──SUBSET_OF── (Deep Learning)
 │                          │
 CREATED_BY              USED_IN
 │                          │
 ▼                          ▼
(Alan Turing)           (TensorFlow)──CREATED_BY──▶(Google)
🧩 How Each Step Works
Step	What It Does	One-Line Explanation
Load	Read faq.txt	Split FAQ into chunks by Q&A pairs
Extract	LLM reads each chunk	Returns entities (nouns) + relations (connections)
Store	Push to Neo4j	Creates nodes + edges in the graph database
Query	User asks question	Find matching graph nodes → send context to LLM → get answer
What is RAG?
Without RAG:  Question → LLM → Answer (guesses from training data)
With RAG:     Question → Search Database → LLM + Context → Accurate Answer ✅
🛠️ Troubleshooting
Problem	Fix
Connection refused	Check Neo4j is running + correct URI in .env
AuthError	Double-check Neo4j password in .env
openai.AuthenticationError	Check your OpenAI API key
Empty results	Make sure you ran the build phase first
🚀 Want to Level Up?
Replace faq.txt with your own company docs or PDFs
Change gpt-4o-mini to gpt-4o for better extraction
Add a Streamlit UI for a web interface
Combine with Vector RAG for hybrid search
