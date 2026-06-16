# Graph Retrieval using the Recipe Dataset

This performs graph search using the recipe dataset.

## Prerequisites

- Python 3.10 **ONLY**

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

Pull and start container images:

```bash
# start podman if you haven't already
podman machine start

# create the network for opensearch
podman network create graph-net

# deploy the container images
podman run -d \
    --name "neo4j-single" \
    --network "graph-net" \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -v "$HOME/neo4j/data:/data" \
    -v "$HOME/neo4j/plugins:/plugins" \
    -v "$HOME/neo4j/import:/var/lib/neo4j/import" \
    neo4j:5.26.16 >/dev/null
```

## Usage

### Step 1: Ingest data for Graph using Neo4j

```bash
python ingest.py
```

### Step 2: Run some sample commands

This demonstrates a simple search using the recipe dataset. We are going to query for simple "noodle" recipes without Sulfites.

```bash
python query_recipes.py --query noodle --exclude-caution "Sulfites"
```

This demonstrates how graph search can factually provide better answers. We want noodle recieps without soba containing no Sulfites.

```bash
python query_recipes.py --query noodle --notquery soba --exclude-caution "Sulfites"
```
