from sentence_transformers import SentenceTransformer, util
# Load the model (0.6B is great for speed, 8B for deep semantic accuracy)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

# 1. Prepare your book data (Documents)
books = [
    "The Great Gatsby by F. Scott Fitzgerald - A story of wealth and obsession in the Jazz Age.",
    "The Hobbit by J.R.R. Tolkien - An adventure of a small hobbit in Middle-earth.",
    "Foundation by Isaac Asimov - A galactic empire faces collapse and rebirth."
]

# 2. Encode Books (No prompt needed for the storage phase)
book_embeddings = model.encode(books)

# 3. Encode Query (MUST use 'query' prompt for retrieval)
user_query = "a science fiction book about space empires"
query_embedding = model.encode(
    user_query, 
    prompt_name="query"  # Crucial: Tells the model this is a search question
)

hits = util.semantic_search(query_embedding, book_embeddings, top_k=2)

# 5. Output the results
print(f"Query: {user_query}\n")
for hit in hits[0]:
    book_index = hit['corpus_id']
    score = hit['score']
    print(f"Result: {books[book_index]} (Similarity Score: {score:.4f})")