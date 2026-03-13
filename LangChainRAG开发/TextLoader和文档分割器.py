from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("./data/小说.txt")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n","\n","。","?","!","？","！"],
    length_function=len,
)

split_docs = splitter.split_documents(docs)

print(len(split_docs))
for doc in split_docs:
    print("=" * 20)
    print(doc)
    print("=" * 20)