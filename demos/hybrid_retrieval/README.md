# Hybrid Retrieval using the Recipe Dataset

This performs hybrid retrieval using the recipe dataset.

## Prerequisites

- Python 3.10 **ONLY**

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

Pull and start container images for Neo4j:

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

Pull and start container images for OpenSearch:

```bash
# start podman if you haven't already
podman machine start

# create the network for opensearch
podman network create opensearch-net

# deploy the container images
podman run -d \
    --name "opensearch-single" \
    --network "opensearch-net" \
    -p 9200:9200 -p 9600:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -e "cluster.routing.allocation.disk.threshold_enabled=false" \
    -v "$HOME/opensearch/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch/snapshots:/mnt/snapshots" \
    opensearchproject/opensearch:3.2.0 >/dev/null

podman run -d \
    --name "opensearch-single-dashboards" \
    --network "opensearch-net" \
    -p 5600:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-single:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.2.0 >/dev/null
```

## Usage

### Step 1: (OPTIONAL) Ingest data for Vector and Graph

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

If you want to see the vector and graph text, run the following command:

```bash
python query_recipes.py --query noodle --notquery soba --exclude-caution "Sulfites" --verbose
```

The sample inference (this requires an inference provide... I am using Nebius):

```bash
python inference.py 
```
