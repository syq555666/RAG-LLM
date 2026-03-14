from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path= "./data/长生渡.pdf",
    mode = "page",            #默认page
    #password = "123456"
)

i=0
for docs in loader.lazy_load():
    i+=1
    print(docs)
    print("="*20)