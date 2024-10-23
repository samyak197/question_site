import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Google Gemini API key
gemini_api_key = "AIzaSyCuFlDA-DbJjBS6o1Ej-rIbKf4ixjzfndw"
genai.configure(api_key=gemini_api_key)

# Configuration
embedding_model_name = "dunzhang/stella_en_400M_v5"
query_prompt_name = (
    "s2p_query"  # Specify the query prompt for the SentenceTransformer model
)


class VectorDatabase:
    def __init__(self, index_path):
        # Load the FAISS index from the saved file
        self.index = faiss.read_index(index_path)

    def query(self, embedding, top_n=5):
        """Query the FAISS vector database to find the most similar embeddings."""
        distances, indices = self.index.search(embedding, top_n)
        return distances, indices


class AIChatbot:
    def __init__(self):
        # Load the embedding model for query encoding (running on CPU for inference)
        self.embedding_model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,
            device="cpu",
            config_kwargs={
                "use_memory_efficient_attention": False,
                "unpad_inputs": False,
            },
        )

        # Load the pre-saved FAISS vector database
        self.vector_db = VectorDatabase("vector_database.faiss")

        # Load text data for retrieving chunks
        file_path = "vit_edu_text.txt"
        with open(file_path, "r", encoding="utf-8") as file:
            self.chunks = (
                file.readlines()
            )  # Each chunk represents a line of the text file

    def encode_text(self, text, prompt_name=None):
        """Encode text into embeddings using the pre-trained SentenceTransformer model."""
        # Use the query prompt for queries
        if prompt_name:
            embeddings = self.embedding_model.encode(text, prompt_name=prompt_name)
        else:
            embeddings = self.embedding_model.encode(text)

        return embeddings

    def retrieve_chunks(self, query, top_n=5):
        """Retrieve and return the most similar chunks based on a query."""
        # Encode the query into an embedding
        query_embedding = self.encode_text(query, prompt_name=query_prompt_name)
        query_embedding = query_embedding.reshape(
            1, -1
        )  # Reshape to 2D for FAISS search

        # Query the vector database for the most similar embeddings
        distances, indices = self.vector_db.query(query_embedding, top_n)

        # Check if the indices are valid and handle cases where no results are found
        if len(indices) == 0 or len(indices[0]) == 0:
            return "No relevant information found in the database."

        # Ensure indices are within bounds and retrieve the chunks
        valid_indices = [index for index in indices[0] if 0 <= index < len(self.chunks)]
        if not valid_indices:
            return "No valid chunks found for the query."

        # Retrieve the most similar chunks based on the valid indices
        retrieved_chunks = " ".join(self.chunks[index] for index in valid_indices)

        return retrieved_chunks

    def generate_response(self, question):
        """Generate a response to a given question using Google Gemini."""
        # Retrieve chunks related to the query
        context = self.retrieve_chunks(question)

        # Define the input template for the Google Gemini API
        input_template = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                Cutting Knowledge Date: December 2023
                Today Date: 2nd October 2024
                <|eot_id|><|start_header_id|>user<|end_header_id|>

                You are a helpful assistant that responds to the user based on the context provided, you can wish or greet the user, if the answer does not lie in the context, you will respond with 'I don't have sufficient information.'

                Context : {context}
                Question : {question}

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """

        # Format the input text using the context and question
        input_text = input_template.format(context=context, question=question)

        # Generate the response using Google Gemini
        model = genai.GenerativeModel("gemini-1.5-pro")

        response = model.generate_content(input_text)
        yield response.text
        # Return the generated response
        # for message in response.text:
        #     yield message.text


# Create an instance of the chatbot and test the response generation
chatbot = AIChatbot()

# Example question to test
question = "What is the fee structure?"

# Generate and print the response
for response_text in chatbot.generate_response(question):
    print(response_text)
