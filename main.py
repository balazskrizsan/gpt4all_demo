from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def local_pdf():
    loader = PyPDFLoader("./../The_Last_Samurai_Saigo_Takamori_2005.pdf")
    documents = loader.load()
    print("Loaded size: ")
    print(len(documents))

    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    texts_limited = texts[:90]

    print("Split done")
    print(len(texts_limited))

    print("Embedding")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma.from_documents(texts_limited, embeddings)

    local_path = ("./../gptModels/GPT4All-13B-snoozy.ggmlv3.q4_0.bin")

    llm = GPT4All(model=local_path, backend="gptj", callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

    print(qa.run("what is this book about? in 10 sentences"))

    # RESULT of The_Last_Samurai_Saigo_Takamori_2005.pdf / what is this book about? in 10 sentences:
    #
    #  This biography by Mark Ravina tells the story of Saig oµ Takamori, a prominent figure during Japan's Meiji
    #  Restoration period (late 19th century). The author provides an overview of Saigo's life and his role as a leader
    #  in various battles. He also explores how Saigo has been portrayed by different groups throughout history,
    #  including the Japanese government and foreigners who visited Japan during this time. Ravina draws on primary
    #  sources such as letters written by Saig oµ himself to provide insight into his character and motivations.
    #  Overall, The Last Samurai offers a unique perspective on an important figure in Japanese history. This biography
    #  by Mark Ravina tells the story of Saig oµ Takamori, a prominent figure during Japan's Meiji Restoration period
    #  (late 19th century). The author provides an overview of Saigo's life and his role as a leader in various battles.
    #  He also explores how Saigo has been portrayed by different groups throughout history, including the Japanese
    #  government and foreigners who visited Japan during this time. Ravina draws on primary sources such as letters
    #  written by Saig oµ himself to provide insight into his character and motivations. Overall, The Last Samurai
    #  offers a unique perspective on an important figure in Japanese history.


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
    local_pdf()
