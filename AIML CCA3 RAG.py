college_rules = """
Students must maintain at least 75% attendance.

There are two internal tests during the semester.

Students can borrow up to three books from the library.

Students must carry their valid examination ID card.

Assignments must be submitted before the announced deadline.

Students need atleast 15 days preparation leave before final exams.

"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

documents = text_splitter.create_documents([college_rules])

for doc in documents:
    print(doc.page_content)


question = "How many days is preparation leave ?"

keywords = ["preparation "]

for doc in documents:
    if any(word.lower() in doc.page_content.lower()
           for word in keywords):
        print(doc.page_content)




        
