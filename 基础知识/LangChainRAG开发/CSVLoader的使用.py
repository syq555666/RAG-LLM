from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(
    file_path="./LangChainRAG开发/data/stu.csv",
    csv_args={
        "delimiter": ",",   #指定分隔符
        "quotechar": '"',   #制定带有分隔符文本引号包含的内容
        "fieldnames": ['a','b','c','d']   #表头
    }
)

documents = loader.load()

#批量加载(内存大)
#for document in documents:
#    print(type(document),document)

#懒加载
for document in loader.lazy_load():
    print(document)