from rich.console import Console
from vectordb import VectorDB
from VectorTfIdfDB import VectorTIdfDB
from IVFPQVectorDB import IVFPQVectorDB
from utils import print_in_box, format_similar_doc, get_user_question

API_KEY = ""

if __name__ == "__main__":
    # db = VectorDB(api_key=API_KEY)
    # db = VectorTIdfDB(api_key=API_KEY)
    db = IVFPQVectorDB(api_key=API_KEY)
    db.load_data('data/data.json')
    db.save_db()

    console = Console()
    console.print("[cyan]Welcome![/cyan] Please type your question below:")

    while True:
        question = get_user_question(console=console)
        # response = db.search(question.lower(), k=2, similarity_threshold=0.3, vector_weight=0.6)
        response = db.search(question.lower(), k=3, similarity_threshold=0.4)
        similar_docs = format_similar_doc(response)
        print_in_box(similar_docs, title="Similar Documents", color="green")
