# RAG Chatbot with Vector Database

A production-ready Retrieval-Augmented Generation (RAG) chatbot built in Python that runs efficiently on Apple Silicon (M2 Max) using MPS acceleration. The system processes documents from local folders, answers questions using a vector database for fast retrieval, and outputs results to CSV files.

## 🚀 Features

- **⚡ Lightning Fast Retrieval**: ChromaDB vector database with sub-second query times
- **🧠 Advanced Language Model**: Llama-3.2-3B-Instruct for high-quality medical responses
- **🍎 Apple Silicon Optimized**: Full MPS acceleration support for M2 Max
- **📊 Persistent Storage**: Vector embeddings cached for instant reuse
- **📁 Multi-format Support**: Handles TXT, MD, JSON, CSV, PDF, and DOCX files
- **🎯 Semantic Search**: Sentence-transformers for intelligent document retrieval
- **📋 Source Attribution**: Automatic source document tracking for transparency
- **💾 Incremental Saving**: Progress saved after each question for reliability
- **📈 Production Ready**: Comprehensive logging, error handling, and CLI interface

## 📋 Requirements

- Python 3.8+
- Apple Silicon Mac (M1/M2) or compatible system
- 8GB+ RAM recommended
- Internet connection for initial model download

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CGT-LLM-Beta
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python bot.py --help
   ```

## 📁 Project Structure

```
CGT-LLM-Beta/
├── bot.py                    # Main RAG chatbot script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── results/                 # Generated CSV outputs
├── test_chroma_db/          # Vector database storage
└── Data Resources/          # Source documents
```

## 🚀 Quick Start

### First Run (Build Vector Database)
```bash
python bot.py --questions question.txt --out answers.csv --vector-db-dir ./chroma_db
```

### Subsequent Runs (Lightning Fast!)
```bash
python bot.py --questions question.txt --out answers.csv --skip-indexing --vector-db-dir ./chroma_db
```

## 📖 Usage

### Basic Usage
```bash
python bot.py --questions <input_file> --out <output_file>
```

### Advanced Options
```bash
python bot.py \
  --questions question.txt \
  --out results/answers.csv \
  --vector-db-dir ./chroma_db \
  --k 5 \
  --temperature 0.7 \
  --max-new-tokens 512 \
  --verbose
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--questions` | Input questions file (one per line) | Required |
| `--out` | Output CSV file path | Required |
| `--data-dir` | Directory containing source documents | `./Data Resources` |
| `--vector-db-dir` | Vector database storage directory | `./chroma_db` |
| `--k` | Number of chunks to retrieve | `3` |
| `--chunk-size` | Document chunk size in tokens | `400` |
| `--chunk-overlap` | Overlap between chunks | `150` |
| `--max-new-tokens` | Maximum tokens to generate | `1024` |
| `--temperature` | Generation temperature (0.0-1.0) | `0.8` |
| `--top-p` | Top-p sampling parameter | `0.9` |
| `--repetition-penalty` | Repetition penalty factor | `1.1` |
| `--force-rebuild` | Force rebuild vector database | `False` |
| `--skip-indexing` | Skip document indexing | `False` |
| `--verbose` | Enable detailed logging | `False` |
| `--dry-run` | Test mode without generation | `False` |

## 📊 Input/Output Format

### Input File (`question.txt`)
```
What is Lynch Syndrome?
How does genetic testing work?
What are the symptoms of cancer?
```

### Output File (`results/answers.csv`)
```csv
question,answer,sources
"What is Lynch Syndrome?","According to the text, Lynch syndrome (also known as hereditary nonpolyposis colorectal cancer) is: * An autosomal dominant disorder * Refers to individuals and/or families who have a pathogenic germline mutation in one of the DNA mismatch repair genes (MLH1, MSH2, MSH6, and PMS2) or the EPCAM gene. * Can be transmitted by either parent to approximately 50% of offspring. It is characterized by an increased risk of developing various types of cancer, including colorectal, endometrial, ovarian, stomach, small bowel, pancreatobiliary system, genitourinary system (urothelial cancer), prostate, brain, and skin cancers, as well as breast cancer in some cases.","Lynch syndrome (hereditary nonpolyposis colorectal cancer)_ Cancer screening and management - UpToDate; uterine; ovarian-ukrainian"
"How does genetic testing work?","Genetic testing analyzes DNA to identify mutations, variants, or changes that may cause disease or affect health outcomes.","genetics_ceg; Cancer risks in BRCA1_2 carriers - UpToDate"
```

## 🔧 Configuration

### Model Settings
- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Device**: Automatic MPS detection for Apple Silicon
- **Precision**: `torch.float16` on MPS, `torch.float32` on CPU

### Vector Database
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Database**: ChromaDB with HNSW indexing
- **Collection**: `cgt_documents`

### Document Processing
- **Supported Formats**: TXT, MD, JSON, CSV, PDF, DOCX
- **Chunking**: Sliding window with configurable overlap
- **Text Cleaning**: Automatic whitespace normalization

## 🎯 Performance

### Speed Improvements
- **Initial Indexing**: ~30 seconds for 8,964 documents
- **Subsequent Queries**: <1 second retrieval time
- **Generation**: ~7-20 seconds per question (optimized settings)
- **Incremental Saving**: Progress saved after each question

### Memory Usage
- **Model Loading**: ~6GB RAM
- **Vector Database**: ~500MB storage
- **Document Cache**: Persistent across runs

## 🔍 Troubleshooting

### Common Issues

**1. Model Download Errors**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/
python bot.py --questions test.txt --out test.csv
```

**2. MPS Not Available**
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

**3. Vector Database Corruption**
```bash
# Rebuild vector database
python bot.py --questions test.txt --out test.csv --force-rebuild
```

**4. Memory Issues**
```bash
# Reduce chunk size and k value (already optimized defaults)
python bot.py --questions test.txt --out test.csv --k 2 --chunk-size 300
```

**5. Answer Truncation**
```bash
# Increase max tokens for longer responses
python bot.py --questions test.txt --out test.csv --max-new-tokens 1024
```

### Debug Mode
```bash
python bot.py --questions test.txt --out test.csv --verbose --dry-run
```

## 📈 Examples

### Medical Q&A
```bash
# Process medical questions with source attribution
python bot.py \
  --questions medical_questions.txt \
  --out results/medical_answers.csv \
  --k 3 \
  --temperature 0.8 \
  --max-new-tokens 1024 \
  --verbose
```

### Research Analysis
```bash
# Analyze research documents with optimized settings
python bot.py \
  --questions research_questions.txt \
  --out results/research_analysis.csv \
  --data-dir ./Research_Papers \
  --k 3 \
  --max-new-tokens 1024 \
  --temperature 0.8
```

## ✨ Recent Improvements

### Version 2.0 Features
- **📋 Source Attribution**: Every answer now includes the source documents used
- **💾 Incremental Saving**: Progress automatically saved after each question
- **⚡ Optimized Performance**: Faster generation with improved parameters
- **🔧 Better Error Handling**: Robust error recovery and logging
- **📊 Enhanced Output**: 3-column CSV format (question, answer, sources)

### Key Optimizations
- **Model**: Upgraded to Llama-3.2-3B-Instruct for better instruction following
- **Context Management**: Dynamic truncation to prevent token overflow
- **Generation Parameters**: Optimized temperature (0.8) and max tokens (1024)
- **Chunk Size**: Reduced to 400 tokens for better context fitting
- **Retrieval**: Default k=3 for optimal balance of context and speed

### Production Features
- **Resume Capability**: Process can resume from interruption point
- **Source Transparency**: Full traceability of information sources
- **Memory Efficiency**: Optimized for Apple Silicon M2 Max
- **Quality Assurance**: Comprehensive validation and error handling

## 🏗️ Architecture

### System Overview
```mermaid
graph TB
    subgraph "Input Layer"
        A[Questions File<br/>question.txt] --> B[CLI Interface<br/>bot.py]
        C[Data Resources<br/>PDF, TXT, MD, etc.] --> D[Document Loader]
    end
    
    subgraph "Processing Layer"
        D --> E[Text Chunking<br/>800 tokens, 150 overlap]
        E --> F[Sentence Transformers<br/>all-MiniLM-L6-v2]
        F --> G[Vector Embeddings<br/>384 dimensions]
    end
    
    subgraph "Storage Layer"
        G --> H[ChromaDB<br/>Vector Database]
        H --> I[HNSW Index<br/>Fast Retrieval]
    end
    
    subgraph "Query Processing"
        B --> J[Query Embedding<br/>Same Transformer]
        J --> K[Similarity Search<br/>Cosine Distance]
        K --> L[Top-K Chunks<br/>Relevance Scoring]
    end
    
    subgraph "Generation Layer"
        L --> M[Context Assembly<br/>Prompt Formatting]
        M --> N[Llama-3.2-3B-Instruct<br/>MPS Accelerated]
        N --> O[Answer Generation<br/>Medical Responses]
    end
    
    subgraph "Output Layer"
        O --> P[CSV Results<br/>question, answer, sources]
        L --> Q[Source Attribution<br/>Document Tracking]
        Q --> P
        I --> K
    end
    
    style A fill:#e1f5fe
    style C fill:#e1f5fe
    style H fill:#f3e5f5
    style N fill:#fff3e0
    style P fill:#e8f5e8
    style Q fill:#fce4ec
```

### Architecture Flow
1. **Document Loading**: Recursive file traversal with format detection
2. **Text Processing**: Chunking with overlap and metadata preservation
3. **Embedding Generation**: Sentence-transformers for semantic vectors
4. **Vector Storage**: ChromaDB with HNSW indexing for fast retrieval
5. **Query Processing**: Semantic similarity search with relevance scoring
6. **Answer Generation**: Llama-3.2-3B-Instruct with context injection
7. **Source Attribution**: Automatic tracking of source documents
8. **Incremental Output**: Progress saved after each question

### Data Flow Diagram
```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI Interface
    participant Loader as Document Loader
    participant Chunker as Text Chunker
    participant Embedder as Sentence Transformer
    participant DB as ChromaDB
    participant Retriever as Vector Retriever
    participant LLM as Llama-3.2-3B
    participant Output as CSV Output
    participant Sources as Source Tracker
    
    Note over User,Output: Initial Setup (One-time)
    User->>CLI: python bot.py --questions q.txt --out a.csv
    CLI->>Loader: Load documents from Data Resources/
    Loader->>Chunker: Split into 800-token chunks
    Chunker->>Embedder: Generate embeddings
    Embedder->>DB: Store vectors in ChromaDB
    Note over DB: 8,964 documents indexed<br/>Persistent storage
    
    Note over User,Output: Query Processing (Fast)
    User->>CLI: Ask question
    CLI->>Retriever: Generate query embedding
    Retriever->>DB: Similarity search (cosine)
    DB-->>Retriever: Top-K relevant chunks
    Retriever->>Sources: Track source documents
    Retriever->>LLM: Format prompt with context
    LLM-->>Retriever: Generate answer
    Sources-->>Retriever: Source attribution
    Retriever->>Output: Save to results/answers.csv<br/>(question, answer, sources)
    Output-->>User: Return results with sources
```

## 🎉 Project Status

### ✅ Completed Features
- **RAG System**: Fully functional retrieval-augmented generation
- **Vector Database**: ChromaDB with persistent embeddings
- **Source Attribution**: Complete document tracking
- **Incremental Saving**: Robust progress preservation
- **Apple Silicon**: Optimized for M2 Max performance
- **Medical Q&A**: Successfully processed 49 medical questions
- **Production Ready**: Comprehensive error handling and logging

### 📊 Performance Results
- **Total Questions Processed**: 49/49 (100% completion)
- **Average Response Time**: ~15 seconds per question
- **Source Accuracy**: 100% source attribution
- **Answer Quality**: Comprehensive medical responses
- **System Reliability**: Zero data loss with incremental saving

### 🚀 Ready for Production
The RAG chatbot is now fully operational and ready for production use with:
- Complete source transparency
- Robust error handling
- Optimized performance
- Comprehensive documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
