# Vector Retrieval using the Recipe Dataset

This performs vector search using the recipe dataset.

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

### Step 1: Ingest data for Vector embeddings on OpenSearch

```bash
python ingest.py
```

### Step 2: Run some sample commands

This demonstrates a simple search using the recipe dataset. We are going to query for simple "noodle" recipes.

```bash
python query_recipes.py --query "I am looking for noodle recipes."
```

This demonstrates how vector search doesn't do well with "negated search".

```bash
python query_recipes.py --query "I am looking for noodles recipes without soba that doesn't contain Sulfites."
```
