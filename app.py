import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# ─────────────────────────────────────────────
# LOAD ENVIRONMENT
# ─────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("Using Neo4j URI:", NEO4J_URI)

if not all([OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("❌ Missing environment variables. Check your .env file.")

# ─────────────────────────────────────────────
# INITIALIZE CLIENTS
# ─────────────────────────────────────────────
ai = OpenAI(api_key=OPENAI_API_KEY)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# ─────────────────────────────────────────────
# CONNECTION TEST
# ─────────────────────────────────────────────
def test_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            print("✅ Neo4j Connection Successful:", result.single())
    except Exception as e:
        print("❌ Neo4j Connection Failed")
        raise e

# ─────────────────────────────────────────────
# STEP 1: LOAD FAQ
# ─────────────────────────────────────────────
def load_faq(path="faq.txt"):
    with open(path, "r") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    print(f"📄 Loaded {len(chunks)} chunks")
    return chunks

# ─────────────────────────────────────────────
# STEP 2: EXTRACT ENTITIES + RELATIONS
# ─────────────────────────────────────────────
def extract(chunk):
    prompt = f"""
Extract entities and relations from this text.

Return ONLY valid JSON:
{{
  "entities": [{{"name": "AI", "type": "CONCEPT"}}],
  "relations": [{{"source": "ML", "relation": "SUBSET_OF", "target": "AI"}}]
}}

Text:
{chunk}
"""

    response = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    try:
        return json.loads(text)
    except:
        return {"entities": [], "relations": []}

# ─────────────────────────────────────────────
# STEP 3: STORE IN AURA
# ─────────────────────────────────────────────
def store(entities, relations):
    with driver.session() as session:

        # Clear old data
        session.run("MATCH (n) DETACH DELETE n")

        # Insert nodes
        for e in entities:
            session.run("""
                MERGE (n:Entity {name: $name})
                SET n.type = $type
            """, name=e["name"], type=e["type"])

        # Insert relationships
        for r in relations:
            session.run("""
                MATCH (a:Entity {name: $src})
                MATCH (b:Entity {name: $tgt})
                MERGE (a)-[:RELATES {type: $rel}]->(b)
            """, src=r["source"], rel=r["relation"], tgt=r["target"])

    print(f"💾 Stored {len(entities)} entities, {len(relations)} relations")

# ─────────────────────────────────────────────
# STEP 4: SEARCH GRAPH
# ─────────────────────────────────────────────
def search_graph(question):

    response = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f'Extract keywords from this question as JSON array: "{question}"'
        }],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    try:
        keywords = json.loads(text)
    except:
        keywords = []

    results = []

    with driver.session() as session:
        for kw in keywords:
            records = session.run("""
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($kw)
                OPTIONAL MATCH (n)-[r:RELATES]->(m)
                OPTIONAL MATCH (p)-[r2:RELATES]->(n)
                RETURN n.name AS entity,
                       n.type AS type,
                       collect(DISTINCT {rel: r.type, target: m.name}) AS out,
                       collect(DISTINCT {rel: r2.type, source: p.name}) AS inc
            """, kw=kw)

            for record in records:
                data = record.data()
                info = f"{data['entity']} ({data['type']})"

                for o in data["out"]:
                    if o["target"]:
                        info += f"\n  → {data['entity']} --{o['rel']}--> {o['target']}"

                for i in data["inc"]:
                    if i["source"]:
                        info += f"\n  ← {i['source']} --{i['rel']}--> {data['entity']}"

                results.append(info)

    return "\n\n".join(results) if results else "No info found."

# ─────────────────────────────────────────────
# STEP 5: ASK
# ─────────────────────────────────────────────
def ask(question):
    context = search_graph(question)

    response = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer strictly using only the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    test_connection()

    print("\n🚀 Building Knowledge Graph...\n")

    chunks = load_faq()

    all_entities = []
    all_relations = []
    seen_entities = set()
    seen_relations = set()

    for i, chunk in enumerate(chunks):
        print(f"  🧠 Extracting chunk {i+1}/{len(chunks)}...")
        data = extract(chunk)

        for e in data["entities"]:
            if e["name"] not in seen_entities:
                seen_entities.add(e["name"])
                all_entities.append(e)

        for r in data["relations"]:
            key = (r["source"], r["relation"], r["target"])
            if key not in seen_relations:
                seen_relations.add(key)
                all_relations.append(r)

    store(all_entities, all_relations)

    print("\n✅ Knowledge Graph ready!")
    print("\n💬 Ask questions (type 'quit' to exit)\n")

    while True:
        q = input("❓ Question: ").strip()
        if q.lower() in ["quit", "exit", "q"]:
            break
        if q:
            print("\n💡", ask(q), "\n")

    driver.close()
