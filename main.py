import os

from ir.vectorspace import VectorSpace

def main():
    vs = VectorSpace()
    files_path = os.path.join(os.path.dirname(__file__), "data", "EnglishNews")
    vs.build(documents_directory=files_path, sample_size=1000, to_sort=True)
    # print(vs.documents_name_content)
    # print(vs.documents_name_content["News996.txt"])
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[0]]))
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[100]]))
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[200]]))
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[-1]]))
    # print(type(name_content["News996.txt"]))
    # parser = Parser()


if __name__ == "__main__":
    main()
