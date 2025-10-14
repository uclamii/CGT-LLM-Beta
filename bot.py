#!/usr/bin/env python3
"""
RAG Chatbot Implementation for CGT-LLM-Beta with Vector Database
Production-ready local RAG system with ChromaDB and MPS acceleration for Apple Silicon
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import textstat

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional imports with graceful fallbacks
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pypdf not available. PDF files will be skipped.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not available. DOCX files will be skipped.")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. BM25 retrieval disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_bot.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with metadata"""
    filename: str
    content: str
    filepath: str
    file_type: str
    chunk_count: int = 0
    file_hash: str = ""


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    filename: str
    chunk_id: int
    total_chunks: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    chunk_hash: str = ""


class VectorRetriever:
    """ChromaDB-based vector retrieval"""
    
    def __init__(self, collection_name: str = "cgt_documents", persist_directory: str = "./chroma_db"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required for vector retrieval")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection '{collection_name}' with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "CGT-LLM-Beta document collection"}
            )
            logger.info(f"Created new collection '{collection_name}'")
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers embedding model")
        else:
            self.embedding_model = None
            logger.warning("Sentence-transformers not available, using ChromaDB default embeddings")
    
    def add_documents(self, chunks: List[Chunk]) -> None:
        """Add document chunks to the vector database"""
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector database...")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            chunk_id = f"{chunk.filename}_{chunk.chunk_id}"
            documents.append(chunk.text)
            
            metadata = {
                "filename": chunk.filename,
                "chunk_id": chunk.chunk_id,
                "total_chunks": chunk.total_chunks,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "chunk_hash": chunk.chunk_hash,
                **chunk.metadata
            }
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        # Add to collection
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(chunks)} chunks to vector database")
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity"""
        try:
            # Perform vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            chunks_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    chunk = Chunk(
                        text=doc,
                        filename=metadata['filename'],
                        chunk_id=metadata['chunk_id'],
                        total_chunks=metadata['total_chunks'],
                        start_pos=metadata['start_pos'],
                        end_pos=metadata['end_pos'],
                        metadata={k: v for k, v in metadata.items() 
                                if k not in ['filename', 'chunk_id', 'total_chunks', 'start_pos', 'end_pos', 'chunk_hash']},
                        chunk_hash=metadata.get('chunk_hash', '')
                    )
                    chunks_with_scores.append((chunk, similarity_score))
            
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


class RAGBot:
    """Main RAG chatbot class with vector database"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.vector_retriever = None
        
        # Load model
        self._load_model()
        
        # Initialize vector retriever
        self._setup_vector_retriever()
    
    def _setup_device(self) -> str:
        """Setup device with MPS support for Apple Silicon"""
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using device: mps (Apple Silicon)")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("Using device: cuda")
        else:
            device = "cpu"
            logger.info("Using device: cpu")
        
        return device
    
    def _load_model(self):
        """Load the Llama model and tokenizer"""
        try:
            logger.info("Loading Llama-3.2-3B-Instruct model...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(2)
    
    def _setup_vector_retriever(self):
        """Setup the vector retriever"""
        try:
            self.vector_retriever = VectorRetriever(
                collection_name="cgt_documents",
                persist_directory=self.args.vector_db_dir
            )
            logger.info("Vector retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup vector retriever: {e}")
            sys.exit(2)
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate hash of file for change detection"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _calculate_chunk_hash(self, text: str) -> str:
        """Calculate hash of chunk text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def load_corpus(self, data_dir: str) -> List[Document]:
        """Load all documents from the data directory"""
        logger.info(f"Loading corpus from {data_dir}")
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Data directory {data_dir} does not exist")
            sys.exit(1)
        
        # Supported file extensions
        supported_extensions = {'.txt', '.md', '.json', '.csv'}
        if PDF_AVAILABLE:
            supported_extensions.add('.pdf')
        if DOCX_AVAILABLE:
            supported_extensions.add('.docx')
            supported_extensions.add('.doc')
        
        # Find all files recursively
        files = []
        for ext in supported_extensions:
            files.extend(data_path.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files with progress bar
        for file_path in tqdm(files, desc="Loading documents"):
            try:
                content = self._read_file(file_path)
                if content.strip():  # Only add non-empty documents
                    file_hash = self._calculate_file_hash(file_path)
                    doc = Document(
                        filename=file_path.name,
                        content=content,
                        filepath=str(file_path),
                        file_type=file_path.suffix.lower(),
                        file_hash=file_hash
                    )
                    documents.append(doc)
                    logger.debug(f"Loaded {file_path.name} ({len(content)} chars)")
                else:
                    logger.warning(f"Skipping empty file: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _read_file(self, file_path: Path) -> str:
        """Read content from various file types"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                return file_path.read_text(encoding='utf-8')
            
            elif suffix == '.md':
                return file_path.read_text(encoding='utf-8')
            
            elif suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return json.dumps(data, indent=2)
                    else:
                        return str(data)
            
            elif suffix == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
            
            elif suffix == '.pdf' and PDF_AVAILABLE:
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            elif suffix in ['.docx', '.doc'] and DOCX_AVAILABLE:
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return ""
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    def chunk_documents(self, docs: List[Document], chunk_size: int, overlap: int) -> List[Chunk]:
        """Chunk documents into smaller pieces"""
        logger.info(f"Chunking {len(docs)} documents (size={chunk_size}, overlap={overlap})")
        chunks = []
        
        for doc in docs:
            doc_chunks = self._chunk_text(
                doc.content, 
                doc.filename, 
                chunk_size, 
                overlap
            )
            chunks.extend(doc_chunks)
            
            # Update document metadata
            doc.chunk_count = len(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
        return chunks
    
    def _chunk_text(self, text: str, filename: str, chunk_size: int, overlap: int) -> List[Chunk]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Simple token-based chunking (approximate)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunk_hash = self._calculate_chunk_hash(chunk_text)
                chunk = Chunk(
                    text=chunk_text,
                    filename=filename,
                    chunk_id=len(chunks),
                    total_chunks=0,  # Will be updated later
                    start_pos=i,
                    end_pos=i + len(chunk_words),
                    metadata={
                        'word_count': len(chunk_words),
                        'char_count': len(chunk_text)
                    },
                    chunk_hash=chunk_hash
                )
                chunks.append(chunk)
        
        # Update total_chunks for each chunk
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def build_or_update_index(self, chunks: List[Chunk], force_rebuild: bool = False) -> None:
        """Build or update the vector index"""
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return
        
        # Check if we need to rebuild
        collection_stats = self.vector_retriever.get_collection_stats()
        existing_count = collection_stats.get('total_chunks', 0)
        
        if existing_count > 0 and not force_rebuild:
            logger.info(f"Vector database already contains {existing_count} chunks. Use --force-rebuild to rebuild.")
            return
        
        if force_rebuild and existing_count > 0:
            logger.info("Force rebuild requested. Clearing existing collection...")
            try:
                self.client.delete_collection(self.vector_retriever.collection_name)
                self.vector_retriever.collection = self.client.create_collection(
                    name=self.vector_retriever.collection_name,
                    metadata={"description": "CGT-LLM-Beta document collection"}
                )
            except Exception as e:
                logger.error(f"Error clearing collection: {e}")
        
        # Add chunks to vector database
        self.vector_retriever.add_documents(chunks)
        
        logger.info("Vector index built successfully")
    
    def retrieve(self, query: str, k: int) -> List[Chunk]:
        """Retrieve relevant chunks for a query using vector search"""
        results = self.vector_retriever.search(query, k)
        chunks = [chunk for chunk, score in results]
        
        if self.args.verbose:
            logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
            for i, (chunk, score) in enumerate(results):
                logger.info(f"  {i+1}. {chunk.filename} (score: {score:.3f})")
        
        return chunks
    
    def format_prompt(self, context_chunks: List[Chunk], question: str) -> str:
        """Format the prompt with context and question, ensuring it fits within token limits"""
        context_parts = []
        for chunk in context_chunks:
            context_parts.append(f"{chunk.text}")
        
        context = "\n".join(context_parts)
        
        # Create base prompt structure
        base_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant. Answer questions based on the provided context. Be specific and informative.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Check if prompt is too long and truncate context if needed
        max_context_tokens = 1200  # Leave room for generation
        tokenized = self.tokenizer(base_prompt, return_tensors="pt")
        current_tokens = tokenized['input_ids'].shape[1]
        
        if current_tokens > max_context_tokens:
            # Truncate context to fit within limits
            context_tokens = self.tokenizer(context, return_tensors="pt")['input_ids'].shape[1]
            available_tokens = max_context_tokens - (current_tokens - context_tokens)
            
            if available_tokens > 0:
                # Truncate context to fit
                truncated_context = self.tokenizer.decode(
                    self.tokenizer(context, return_tensors="pt", truncation=True, max_length=available_tokens)['input_ids'][0],
                    skip_special_tokens=True
                )
                
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant. Answer questions based on the provided context. Be specific and informative.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {truncated_context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            else:
                # If even basic prompt is too long, use minimal format
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Answer the question based on the context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context[:500]}...

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            prompt = base_prompt
            
        return prompt
    
    def generate_answer(self, prompt: str, **gen_kwargs) -> str:
        """Generate answer using the language model"""
        try:
            if self.args.verbose:
                logger.info(f"Full prompt (first 500 chars): {prompt[:500]}...")
            
            # Tokenize input with more conservative limit to leave room for generation
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.args.verbose:
                logger.info(f"Input tokens: {inputs['input_ids'].shape}")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get('max_new_tokens', 512),
                    temperature=gen_kwargs.get('temperature', 0.7),
                    top_p=gen_kwargs.get('top_p', 0.95),
                    repetition_penalty=gen_kwargs.get('repetition_penalty', 1.05),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
            
            # Decode response without skipping special tokens to preserve full length
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            if self.args.verbose:
                logger.info(f"Full response (first 1000 chars): {response[:1000]}...")
                logger.info(f"Looking for 'Answer:' in response: {'Answer:' in response}")
                if "Answer:" in response:
                    answer_part = response.split("Answer:")[-1]
                    logger.info(f"Answer part (first 200 chars): {answer_part[:200]}...")
                
                # Debug: Show the full response to understand the structure
                logger.info(f"Full response length: {len(response)}")
                logger.info(f"Prompt length: {len(prompt)}")
                logger.info(f"Response after prompt (first 500 chars): {response[len(prompt):][:500]}...")
            
            # Extract the answer more robustly by looking for the end of the prompt
            # Find the actual end of the prompt in the response
            prompt_end_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if prompt_end_marker in response:
                answer = response.split(prompt_end_marker)[-1].strip()
            else:
                # Fallback to character-based extraction
                answer = response[len(prompt):].strip()
            
            if self.args.verbose:
                logger.info(f"Full LLM output (first 200 chars): {answer[:200]}...")
                logger.info(f"Full LLM output length: {len(answer)} characters")
                logger.info(f"Full LLM output (last 200 chars): ...{answer[-200:]}")
            
            # Only do minimal cleanup to preserve the full response
            # Remove special tokens that might interfere with display, but preserve content
            if "<|start_header_id|>" in answer:
                # Only remove if it's at the very end
                if answer.endswith("<|start_header_id|>"):
                    answer = answer[:-len("<|start_header_id|>")].strip()
            if "<|eot_id|>" in answer:
                # Only remove if it's at the very end
                if answer.endswith("<|eot_id|>"):
                    answer = answer[:-len("<|eot_id|>")].strip()
            if "<|end_of_text|>" in answer:
                # Only remove if it's at the very end
                if answer.endswith("<|end_of_text|>"):
                    answer = answer[:-len("<|end_of_text|>")].strip()
            
            # Final validation - only reject if completely empty
            if not answer or len(answer) < 3:
                answer = "I don't know."
            
            if self.args.verbose:
                logger.info(f"Final answer: '{answer}'")
            
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I encountered an error while generating the answer."
    
    def process_questions(self, questions_path: str, **kwargs) -> List[Tuple[str, str, str, str, float]]:
        """Process all questions and generate answers"""
        logger.info(f"Processing questions from {questions_path}")
        
        # Load questions
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load questions: {e}")
            sys.exit(1)
        
        logger.info(f"Found {len(questions)} questions to process")
        
        qa_pairs = []
        
        # Initialize CSV file with headers
        self.write_csv([], kwargs.get('output_file', 'results.csv'), append=False)
        
        # Process each question
        for i, question in enumerate(tqdm(questions, desc="Processing questions")):
            logger.info(f"Question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Retrieve relevant chunks
                context_chunks = self.retrieve(question, self.args.k)
                
                if not context_chunks:
                    answer = "I don't know."
                    sources = "No sources found"
                    simplified_answer = "I don't know."
                    grade_level = 6.0
                else:
                    # Format prompt
                    prompt = self.format_prompt(context_chunks, question)
                    
                    # Generate answer
                    start_time = time.time()
                    answer = self.generate_answer(prompt, **kwargs)
                    gen_time = time.time() - start_time
                    
                    # Extract source documents
                    sources = self._extract_sources(context_chunks)
                    
                    # Enhance readability
                    readability_start = time.time()
                    simplified_answer, grade_level = self.enhance_readability(answer)
                    readability_time = time.time() - readability_start
                    
                    logger.info(f"Generated answer in {gen_time:.2f}s")
                    logger.info(f"Enhanced readability in {readability_time:.2f}s")
                    logger.info(f"Sources: {sources}")
                    logger.info(f"Grade level: {grade_level:.1f}")
                
                qa_pairs.append((question, answer, sources, simplified_answer, grade_level))
                
                # Write incrementally to CSV after each question
                self.write_csv([(question, answer, sources, simplified_answer, grade_level)], kwargs.get('output_file', 'results.csv'), append=True)
                logger.info(f"Progress saved: {i+1}/{len(questions)} questions completed")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                error_answer = "I encountered an error processing this question."
                sources = "Error retrieving sources"
                simplified_answer = "I encountered an error processing this question."
                grade_level = 6.0
                qa_pairs.append((question, error_answer, sources, simplified_answer, grade_level))
                
                # Still write the error to CSV
                self.write_csv([(question, error_answer, sources, simplified_answer, grade_level)], kwargs.get('output_file', 'results.csv'), append=True)
                logger.info(f"Error saved: {i+1}/{len(questions)} questions completed")
        
        return qa_pairs
    
    def _extract_sources(self, context_chunks: List[Chunk]) -> str:
        """Extract source document names from context chunks"""
        sources = []
        for chunk in context_chunks:
            # Debug: Print chunk filename if verbose
            if self.args.verbose:
                logger.info(f"Chunk filename: {chunk.filename}")
            
            # Extract filename from chunk attribute (not metadata)
            source = chunk.filename if hasattr(chunk, 'filename') and chunk.filename else 'Unknown source'
            # Clean up the source name
            if source.endswith('.pdf'):
                source = source[:-4]  # Remove .pdf extension
            elif source.endswith('.txt'):
                source = source[:-4]  # Remove .txt extension
            elif source.endswith('.md'):
                source = source[:-3]  # Remove .md extension
            
            sources.append(source)
        
        # Remove duplicates while preserving order
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)
        
        return "; ".join(unique_sources)
    
    def enhance_readability(self, answer: str) -> Tuple[str, float]:
        """Enhance answer readability to 6th grade level and calculate Flesch-Kincaid Grade Level"""
        try:
            # Create a prompt to simplify the medical answer to 6th grade level
            readability_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant who specializes in explaining complex medical information in simple, easy-to-understand language for 6th grade reading level. Rewrite the following medical answer using:
- Simple, everyday words instead of medical jargon
- Shorter sentences
- Clear explanations
- Avoid complex medical terms when possible
- Keep the same important information but make it much easier to read

Target: 6th grade reading level (ages 11-12)<|eot_id|><|start_header_id|>user<|end_header_id|>

Please rewrite this medical answer in simple language for a 6th grader:

{answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
            # Generate simplified answer
            inputs = self.tokenizer(readability_prompt, return_tensors="pt", truncation=True, max_length=2048)
            if self.device == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Shorter for simplified version
                    temperature=0.3,     # Lower temperature for more consistent simplification
                    top_p=0.9,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract simplified answer
            prompt_end_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if prompt_end_marker in response:
                simplified_answer = response.split(prompt_end_marker)[-1].strip()
            else:
                simplified_answer = response[len(readability_prompt):].strip()
            
            # Clean up special tokens
            if "<|eot_id|>" in simplified_answer:
                if simplified_answer.endswith("<|eot_id|>"):
                    simplified_answer = simplified_answer[:-len("<|eot_id|>")].strip()
            if "<|end_of_text|>" in simplified_answer:
                if simplified_answer.endswith("<|end_of_text|>"):
                    simplified_answer = simplified_answer[:-len("<|end_of_text|>")].strip()
            
            # Calculate Flesch-Kincaid Grade Level
            try:
                grade_level = textstat.flesch_kincaid_grade(simplified_answer)
            except:
                grade_level = 0.0
            
            if self.args.verbose:
                logger.info(f"Simplified answer length: {len(simplified_answer)} characters")
                logger.info(f"Flesch-Kincaid Grade Level: {grade_level:.1f}")
            
            return simplified_answer, grade_level
            
        except Exception as e:
            logger.error(f"Error enhancing readability: {e}")
            # Fallback: return original answer with estimated grade level
            try:
                grade_level = textstat.flesch_kincaid_grade(answer)
            except:
                grade_level = 12.0  # Default to high school level
            return answer, grade_level
    
    def write_csv(self, qa_pairs: List[Tuple[str, str, str, str, float]], output_path: str, append: bool = False) -> None:
        """Write Q&A pairs to CSV file in results folder"""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # If output_path doesn't already have results/ prefix, add it
        if not output_path.startswith('results/'):
            output_path = f'results/{output_path}'
        
        if append:
            logger.info(f"Appending results to {output_path}")
        else:
            logger.info(f"Writing results to {output_path}")
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check if file exists and if we're appending
            file_exists = output_path.exists()
            write_mode = 'a' if append and file_exists else 'w'
            
            with open(output_path, write_mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header only if creating new file or first append
                if not append or not file_exists:
                    writer.writerow(['question', 'answer', 'sources', '6th_grade_answer', 'flesch_kincaid_grade_level'])
                
                for question, answer, sources, simplified_answer, grade_level in qa_pairs:
                    # Clean and escape the answer for CSV
                    # Replace newlines with spaces and clean up formatting
                    clean_answer = answer.replace('\n', ' ').replace('\r', ' ')
                    # Remove extra whitespace but preserve the full content
                    clean_answer = ' '.join(clean_answer.split())
                    # Escape quotes properly for CSV
                    clean_answer = clean_answer.replace('"', '""')
                    
                    # Clean sources as well
                    clean_sources = sources.replace('\n', ' ').replace('\r', ' ')
                    clean_sources = ' '.join(clean_sources.split())
                    clean_sources = clean_sources.replace('"', '""')
                    
                    # Clean simplified answer
                    clean_simplified = simplified_answer.replace('\n', ' ').replace('\r', ' ')
                    clean_simplified = ' '.join(clean_simplified.split())
                    clean_simplified = clean_simplified.replace('"', '""')
                    
                    # Log the full answer length for debugging
                    if self.args.verbose:
                        logger.info(f"Writing answer length: {len(clean_answer)} characters")
                        logger.info(f"Simplified answer length: {len(clean_simplified)} characters")
                        logger.info(f"Grade level: {grade_level:.1f}")
                        logger.info(f"Answer preview: {clean_answer[:200]}...")
                        logger.info(f"Sources: {clean_sources}")
                    
                    # Use proper CSV quoting - let csv.writer handle the quoting
                    writer.writerow([question, clean_answer, clean_sources, clean_simplified, f"{grade_level:.1f}"])
            
            if append:
                logger.info(f"Appended {len(qa_pairs)} Q&A pairs to {output_path}")
            else:
                logger.info(f"Successfully wrote {len(qa_pairs)} Q&A pairs to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")
            sys.exit(4)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG Chatbot for CGT-LLM-Beta with Vector Database")
    
    # File paths
    parser.add_argument('--data-dir', default='./Data Resources', 
                       help='Directory containing documents to index')
    parser.add_argument('--questions', default='./questions.txt',
                       help='File containing questions (one per line)')
    parser.add_argument('--out', default='./answers.csv',
                       help='Output CSV file for answers')
    parser.add_argument('--vector-db-dir', default='./chroma_db',
                       help='Directory for ChromaDB persistence')
    
    # Retrieval parameters
    parser.add_argument('--k', type=int, default=3,
                       help='Number of chunks to retrieve per question')
    
    # Chunking parameters
    parser.add_argument('--chunk-size', type=int, default=400,
                       help='Size of text chunks in tokens')
    parser.add_argument('--chunk-overlap', type=int, default=150,
                       help='Overlap between chunks in tokens')
    
    # Generation parameters
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                       help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Generation temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                       help='Repetition penalty')
    
    # Database options
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of vector database')
    parser.add_argument('--skip-indexing', action='store_true',
                       help='Skip document indexing, use existing database')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Build index and test retrieval without generation')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting RAG Chatbot with Vector Database")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Initialize bot
        bot = RAGBot(args)
        
        # Check if we should skip indexing
        if not args.skip_indexing:
            # Load and process documents
            documents = bot.load_corpus(args.data_dir)
            if not documents:
                logger.error("No documents found to process")
                sys.exit(3)
            
            # Chunk documents
            chunks = bot.chunk_documents(documents, args.chunk_size, args.chunk_overlap)
            if not chunks:
                logger.error("No chunks created from documents")
                sys.exit(3)
            
            # Build or update index
            bot.build_or_update_index(chunks, args.force_rebuild)
        else:
            logger.info("Skipping document indexing, using existing vector database")
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return
        
        # Process questions
        generation_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'repetition_penalty': args.repetition_penalty
        }
        
        qa_pairs = bot.process_questions(args.questions, output_file=args.out, **generation_kwargs)
        
        logger.info("RAG Chatbot completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()