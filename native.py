#!/usr/bin/env python

# https://github.com/opensearch-project/opensearch-py/blob/main/samples/knn/knn-boolean-filter.py

# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import os
import random

from opensearchpy import OpenSearch, helpers

host = os.getenv('HOST', default='localhost')
port = int(os.getenv('PORT', 9200))

client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=("admin", "admin"),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# check whether an index exists
# index_name = "test-doq-search-*"
index_name = "lofasz"
dimensions = 5

if not client.indices.exists(index_name):
    client.indices.create(
        index_name,
        body={
            "settings": {
                "index.knn": True
            },
            "mappings": {
                "properties": {
                    "values": {
                        "type": "knn_vector",
                        "dimension": dimensions
                    },
                }
            }
        }
    )

# index data
vectors = []
page = 28
for i in range(3000):
    vec = []
    for j in range(dimensions):
        vec.append(round(random.uniform(0, 1), 2))

    vectors.append({
        "_index": index_name,
        "_id": i,
        "values": vec,
        "metadata": {
            "page": page
        }
    })

# bulk index
helpers.bulk(client, vectors)

client.indices.refresh(index=index_name)

# search
page = page
vec = []
for j in range(dimensions):
    vec.append(round(random.uniform(0, 1), 2))
print(f"Searching for {vec} with the '{page}' page ...")

search_query = {
    "query": {
        "bool": {
            "filter": {
                "bool": {
                    "must": [{
                        "term": {
                            "metadata.page": page
                        }
                    }]
                }
            },
            "must": {
                "knn": {
                    "values": {
                        "vector": vec,
                        "k": 5
                    }
                }
            }
        }
    }
}

results = client.search(index=index_name, body=search_query)
for hit in results["hits"]["hits"]:
    print(hit)

# delete index
client.indices.delete(index=index_name)
