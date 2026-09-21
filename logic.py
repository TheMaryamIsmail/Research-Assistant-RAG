import os
import re
import math
from typing import List, Dict
from collections import Counter
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)


class PureRAGManager:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
        # Generative Model client for Question Answering
        self.client = genai.Client(api_key=self.api_key)
        self.documents: List[Dict] = []  # Stores chunks and text metadata
        self.idf_cache: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple text normalizer & tokenizer."""
        return re.findall(r'\w+', text.lower())

    def _split_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Splits document into readable chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def process_pdfs(self, file_paths: List[Dict[str, str]]) -> int:
        """Processes PDFs and extracts text chunks."""
        self.clear()
        total_chunks = 0

        for file_info in file_paths:
            path = file_info["path"]
            doc_name = file_info["filename"]

            reader = PdfReader(path)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                page_num = page_idx + 1
                chunks = self._split_text(page_text)

                for chunk_text in chunks:
                    if not chunk_text.strip():
                        continue

                    tokens = self._tokenize(chunk_text)
                    self.documents.append({
                        "text": chunk_text,
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "tokens": tokens,
                        "tf": Counter(tokens)
                    })
                    total_chunks += 1

        # Calculate Document Frequencies for TF-IDF matching
        self._compute_idf()
        return total_chunks

    def _compute_idf(self):
        """Computes Inverse Document Frequency (IDF) for all tokens across chunks."""
        N = len(self.documents)
        if N == 0:
            return

        doc_counts = Counter()
        for doc in self.documents:
            unique_tokens = set(doc["tokens"])
            for token in unique_tokens:
                doc_counts[token] += 1

        self.idf_cache = {
            token: math.log((N + 1) / (count + 1)) + 1
            for token, count in doc_counts.items()
        }

    def _score_chunk(self, query_tokens: List[str], doc: Dict) -> float:
        """Computes TF-IDF similarity score between query and document chunk."""
        score = 0.0
        query_tf = Counter(query_tokens)

        for token in set(query_tokens):
            if token in doc["tf"]:
                tf_doc = doc["tf"][token]
                tf_query = query_tf[token]
                idf = self.idf_cache.get(token, 1.0)
                score += (tf_doc * idf) * (tf_query * idf)

        return score

    def _retrieve_top_k(self, question: str, top_k: int = 4) -> List[Dict]:
        """Finds most relevant text chunks using pure TF-IDF search."""
        if not self.documents:
            return []

        query_tokens = self._tokenize(question)
        if not query_tokens:
            return []

        scored_docs = []
        for doc in self.documents:
            score = self._score_chunk(query_tokens, doc)
            if score > 0:
                scored_docs.append((score, doc))

        # Sort by best match score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    def query(self, question: str) -> Dict:
        """Queries Gemini LLM using retrieved text chunks."""
        if not self.documents:
            return {
                "answer": "No research papers have been uploaded yet. Please upload a paper first.",
                "sources": []
            }

        top_chunks = self._retrieve_top_k(question, top_k=5)

        # Fallback if no direct keyword match found
        if not top_chunks:
            top_chunks = self.documents[:3]

        context_str = ""
        sources = []
        seen_sources = set()

        for chunk in top_chunks:
            context_str += f"\n--- [Document: {chunk['doc_name']}, Page: {chunk['page_num']}] ---\n{chunk['text']}\n"
            
            source_key = f"{chunk['doc_name']}-pg-{chunk['page_num']}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "document": chunk["doc_name"],
                    "page": chunk["page_num"]
                })

        system_instruction = (
            "You are a strict research paper assistant.\n"
            "Answer the user's question using ONLY the context provided below.\n"
            "If the answer is not contained within the provided context, state clearly: "
            "\"The requested information is not available in the uploaded research papers.\"\n"
            "Do not hallucinate, speculate, or bring in outside knowledge."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
            )

            return {
                "answer": response.text if response.text else "No answer generated.",
                "sources": sources
            }

        except Exception as e:
            return {
                "answer": f"An error occurred while generating response: {str(e)}",
                "sources": []
            }

    def clear(self):
        """Clears stored PDF documents."""
        self.documents = []
        self.idf_cache = {}