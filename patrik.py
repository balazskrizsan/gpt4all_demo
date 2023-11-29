from concurrent.futures import ThreadPoolExecutor
import boto3
from urllib.parse import unquote_plus
from langchain.llms import SagemakerEndpoint
from langchain.vectorstores import FAISS
from timeit import default_timer as timer
from langchain.prompts import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.chains.question_answering import load_qa_chain
from sagemaker_contenthandlers import *

bucket = "content-ai-index-bucket"
s3 = boto3.client("s3")

embeddings = SagemakerEndpointEmbeddings(
    endpoint_name="EmbeddingsModelEndpoint532B921B-tncaYvjFgbWN",
    region_name="us-west-2",
    content_handler=EmbeddingContentHandler(),
)

stopwords = ["<|endoftext|>", "Question:", "Answer:", "</s>"]
llm = SagemakerEndpoint(
    endpoint_name="h2oaih2ogptgmoasst1en2048falcon40bv264D4AE97-NIjTxt12SMnO",
    region_name="us-west-2",
    content_handler=ContentHandler(),
    model_kwargs={
        "temperature": 0.1,
        "top_p": 0.9,
        "max_new_tokens": 1024,
        "do_sample": True,
        "repetition_penalty": 1.03,
        "stopwords": stopwords,
    },
)


def lambda_handler(event, context):
    question =  event['question'] if 'question' in event else "How can I delete a Board book?"

    start = timer()
    indexes = s3.list_objects(Bucket=bucket)
    keys = list(map(lambda item: unquote_plus(item["Key"]), indexes["Contents"]))

    firstKey = keys.pop(0)
    content = load_s3_index_bytes(firstKey)
    db = FAISS.deserialize_from_bytes(serialized=content, embeddings=embeddings)

    print(type(db.index))

    with ThreadPoolExecutor(10) as executor:
        for result in executor.map(load_index, keys):
            db.merge_from(result)

    end = timer()
    print(f"\n> Merging indexes (took {end - start:0.4f} s.):")

    searchValue = embeddings.embed_query(question)
    vectorSearchResults = db.max_marginal_relevance_search_with_score_by_vector(
        searchValue, lambda_mult=1
    )

    document, rank = sorted(vectorSearchResults, key=lambda t: t[1], reverse=False)[0]
    print(f"> Vector search result from {document.metadata['original_file']}\n")
    print(document.page_content)
    prompt_template = """Use the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        llm=llm,
        prompt=prompt,
    )

    answer = chain(
        {"input_documents": [document], "question": question}, return_only_outputs=True
    )
    print("> LLM answer\n")
    print(answer["output_text"])


def load_index(key):
    content = load_s3_index_bytes(key)
    return FAISS.deserialize_from_bytes(serialized=content, embeddings=embeddings)


def load_s3_index_bytes(key):
    s3_response = s3.get_object(Bucket=bucket, Key=key)
    s3_object_body = s3_response.get("Body")
    return s3_object_body.read()
