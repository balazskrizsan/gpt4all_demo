from os import getenv
from llama_index import SimpleDirectoryReader
from llama_index.vector_stores import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)
from llama_index import VectorStoreIndex, StorageContext

# http endpoint for your cluster (opensearch required for vector index usage)
endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
# index to demonstrate the VectorStore impl
idx = getenv("OPENSEARCH_INDEX", "test-doq-search-samurai")
# load some sample data
documents = SimpleDirectoryReader("./../sample_books").load_data()


# OpensearchVectorClient stores text in this field by default
text_field = "content"
# OpensearchVectorClient stores embeddings in this field by default
embedding_field = "embedding"
# OpensearchVectorClient encapsulates logic for a
# single opensearch index with vector search enabled
auth = {"user": "admin", "pass": "admin"}
client = OpensearchVectorClient(
    endpoint, idx, 1536, embedding_field=embedding_field, text_field=text_field,
    # hosts=[{'host': "http://localhost", 'port': "9200"}],
    http_auth=("admin", "admin"),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)
# initialize vector store
vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# initialize an index using our sample data and the client we just created
index = VectorStoreIndex.from_documents(
    documents=documents, storage_context=storage_context
)

# run query
query_engine = index.as_query_engine()
res = query_engine.query("What did the author do growing up?")
print(res.response)

