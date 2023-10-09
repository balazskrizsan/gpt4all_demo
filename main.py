import itertools

import psycopg2
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from pgvector.psycopg2 import register_vector


def get_db_conn():
    dbconn = psycopg2.connect(
        host="localhost", user="admin", password="admin_pass", database="ai", port="55555", connect_timeout=10
    )

    dbconn.set_session(autocommit=True)

    return dbconn

def parse_and_save():
    db_conn = get_db_conn()
    register_vector(db_conn)
    cur = db_conn.cursor()

    loader = PyPDFLoader("./../The_Last_Samurai_Saigo_Takamori_2005.pdf")
    documents = loader.load()
    print("Loaded size: ")
    print(len(documents))

    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    texts_limited = texts[:30]
    texts_raw = [text.page_content for text in texts_limited]

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    dataToSave = list(zip(texts_raw, embeddings.embed_documents(texts_raw), itertools.count()))

    for dts in dataToSave:
        cur.execute("INSERT INTO documents (document_meta_id, page_number, content, content_embeddings) VALUES (%s, %s, %s, %s)", [
            1, dts[2], dts[0], dts[1]
        ])

    CONNECTION_STRING = "postgresql+psycopg2://admin:admin_pass@localhost:55555/ai"

    db = PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="blog_posts",
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STRING)


    # local_path = ("./../gptModels/GPT4All-13B-snoozy.ggmlv3.q4_0.bin")
    #
    # llm = GPT4All(model=local_path, backend="gptj", callbacks=[StreamingStdOutCallbackHandler()], verbose=True, n_threads=8)
    #
    # qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
    #
    # print(qa.run("what is this book about? in 10 sentences"))

    cur.close()
    db_conn.close()

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
    loader = PyPDFLoader("./../The_Last_Samurai_Saigo_Takamori_2005.pdf")
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

    local_path = ("./../gptModels/GPT4All-13B-snoozy.ggmlv3.q4_0.bin")

    llm = GPT4All(model=local_path, backend="gptj", callbacks=[StreamingStdOutCallbackHandler()], verbose=True, n_threads=8)

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

    print(qa.run("what is this book about? in 10 sentences"))

def v1():
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    local_path = ("./../gptModels/GPT4All-13B-snoozy.ggmlv3.q4_0.bin")

    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
    llm = GPT4All(model=local_path, backend="gptj", callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

    print(llm_chain.run(question))


if __name__ == '__main__':
    parse_and_save()
