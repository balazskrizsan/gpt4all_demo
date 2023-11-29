from datetime import datetime

import psycopg2
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, OpenSearchVectorSearch
from pgvector.psycopg2 import register_vector

llm_local_path = ("./../gptModels/GPT4All-13B-snoozy.ggmlv3.q4_0.bin")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
query = "what is this book about? in 5 sentences"


def get_db_conn():
    dbconn = psycopg2.connect(
        host="localhost", user="admin", password="admin_pass", database="ai", port="55555", connect_timeout=10
    )

    dbconn.set_session(autocommit=True)

    return dbconn


def parse_and_save():
    loader = PyPDFLoader("./../sample_books/The_Last_Samurai_Saigo_Takamori_2005.pdf")
    # loader = PyPDFLoader("./../PDFsecurity.pdf")
    docs = loader.load()
    print("Loaded size: ")
    print(len(docs))
    # docs[0].metadata["custom_page_number"] = "1"
    # docs[0].metadata["unique_id"] = "qqq"
    # docs[1].metadata["custom_page_number"] = "2"
    # docs[1].metadata["unique_id"] = "qqq"
    # docs[2].metadata["custom_page_number"] = "3"
    # docs[2].metadata["unique_id"] = "qqq"

    OpenSearchVectorSearch.from_documents(
        docs[:150],
        embeddings,
        opensearch_url="https://localhost:9200",
        http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        # index_name="test-doq-search-pdf-security",
        index_name="test-doq-search-samurai",
    )


def init():
    db_conn = get_db_conn()
    cur = db_conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    register_vector(db_conn)

    cur.execute("SELECT typname FROM pg_type WHERE typname= 'vector';")

    rows = cur.fetchall()

    for row in rows:
        print(row)

    sql = """
            CREATE TABLE IF NOT EXISTS documents (
                id uuid DEFAULT gen_random_uuid (),
                document_meta_id int,
                page_number int,
                content text,
                content_embeddings vector(384)
                );
        """
    cur.execute(sql)
    cur.execute("CREATE INDEX IF NOT EXISTS fulltxt_idx ON documents USING GIN (to_tsvector('english', content));")
    # cur.execute("CREATE INDEX IF NOT EXISTS url_idx ON documents (url);")


def local_pdf():
    loader = PyPDFLoader("./../PDFsecurity.pdf")
    documents = loader.load()
    print("Loaded size: ")
    print(len(documents))

    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    texts_limited = texts[:30]

    print("Split done")
    print(len(texts_limited))

    print("Embedding")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma.from_documents(texts_limited, embeddings)

    llm = GPT4All(model=llm_local_path, backend="gptj", callbacks=[StreamingStdOutCallbackHandler()], verbose=True,
                  n_threads=8)

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever(
        search_kwargs={'filter': {'paper_title': 'GPT-4 Technical Report'}}
    ))

    print(qa.run("what is this book about? in 10 sentences"))


def v1():
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = GPT4All(model=llm_local_path, backend="gptj", callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

    print(llm_chain.run(question))


def search():
    vector_store = OpenSearchVectorSearch(
        # index_name="test-doq-search-pdf-security",
        index_name="test-doq-search-*",
        embedding_function=embeddings,
        opensearch_url="https://localhost:9200",
        http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    llm = GPT4All(
        model=llm_local_path,
        backend="gptj",
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=True,
        n_threads=8
    )

    specific_documents = 28

    boolean_filter = {
        "bool": {
            "must": [{
                "term": {
                    "metadata.page": specific_documents
                }
            }]
        }
    }

    boolean_filter2 = {
        "terms": {
            "metadata.page": specific_documents
        }
    }

    result = vector_store.max_marginal_relevance_search(
        query="what is this book about? in 5 sentences",
        k=4,
        fetch_k=20,
        lambda_mult=0.5,
        boolean_filter=boolean_filter2,
        # vector_field="values"
    )
    print(result)

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(
    #     search_kwargs={"k": 5, "filter": {"metadata.source": "./../PDFsecurity.pdf"}}
    # ), memory=memory)
    # result = qa({"question": "what is this book about? in 5 sentences"})
    #
    # print(result["answer"])

    # filters = MetadaStaFilters(filters=[ExactMatchFilter(key="source", value="./../PDFsecurity.pdf")])
    # qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_store.as_retriever(
    #     # search_kwargs={"filter": {"source": "./../PDFsecurity.pdf"}}
    #     # search_kwargs={"filter": {"metadata.source": "PDFsecurity"}}
    #     # search_kwargs={"filter": {"metadata.source": "*PDFsecurity*"}}
    #     # search_kwargs={"filter": {"source": "PDFsecurity"}}
    #     # search_kwargs={"filter": {"source": "*PDFsecurity*"}}
    #     search_kwargs={"filter": filters}
    #     # search_kwargs={"metadata_filters": filters}
    # ))
    # print(qa.run("what is this book about? in 5 sentences"))


if __name__ == '__main__':
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    # parse_and_save()
    search()

    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
