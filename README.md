# Docling RAG with LangChain on Google Colab

This repository provides a detailed implementation of **Retrieval-Augmented Generation (RAG)** using **Docling** and **LangChain** on Google Colab. The project demonstrates how to process documents, generate embeddings, store and retrieve them in a vector database, and use a Language Model to answer questions based on the retrieved content. An interactive **Gradio** app is included for real-time interaction.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Workflow](#workflow)
6. [Detailed Steps](#detailed-steps)
7. [Example Queries](#example-queries)
8. [Gradio Interface](#gradio-interface)
9. [File Structure](#file-structure)
10. [References](#references)
11. [Author](#author)

---

## Overview

This project integrates **LangChain** with **Docling** to create an RAG pipeline that:
- Loads documents (PDFs, markdown, or plain text).
- Splits documents into chunks for better processing.
- Encodes chunks into embeddings using Hugging Face models.
- Stores embeddings in a **Milvus** vector database.
- Retrieves relevant chunks based on user queries.
- Generates answers using a Hugging Face language model.

---

## Features

1. **Document Loading and Conversion**: Supports multiple file formats and processes them into structured text using `DoclingPDFLoader`.
2. **Text Splitting**: Efficiently splits documents into smaller, manageable chunks using a customizable chunk size and overlap.
3. **Embedding Generation**: Leverages Hugging Face's pre-trained embedding models for dense vector representations.
4. **Vector Storage with Milvus**: Efficiently stores embeddings for scalable and fast retrieval.
5. **Question-Answering with RAG**: Combines retrieval of relevant document chunks and context-based language model generation for accurate answers.
6. **Interactive Interface**: Gradio-based UI for real-time file uploads and query handling.

---

## Requirements

### Python Packages
Install the following Python packages:

```bash
pip install docling docling-core python-dotenv langchain-text-splitters langchain-huggingface langchain-milvus gradio
```

### Hugging Face API Token
Obtain a Hugging Face API token and set it in your Colab or local environment:
- **For Colab**: Use `google.colab.userdata.get('HF_TOKEN')`.
- **For Local Setup**: Use `dotenv` to load the token from a `.env` file.

### External Tools
- **Milvus**: For embedding storage and retrieval.

---

## Setup

### Step 1: Clone the Repository

Clone the repository to your local system or Colab environment:

```bash
git clone https://github.com/your-repo-name/docling_RAG_langchain
cd docling_RAG_langchain
```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

- Add your Hugging Face API token to the `.env` file or use `google.colab.userdata.get('HF_TOKEN')` for Colab.

### Step 4: Launch Jupyter Notebook or Colab

Open the provided notebook `docling_RAG_langchain_colab.ipynb` and follow the step-by-step instructions.

---

## Workflow

The workflow consists of several key steps:

### 1. **Document Loading**
   - Load documents from either local files or URLs using the `DoclingPDFLoader` class.
   - Convert the files into text that can be processed further.

### 2. **Text Splitting**
   - Use `RecursiveCharacterTextSplitter` to split documents into chunks of text.
   - Adjust the `chunk_size` and `chunk_overlap` parameters for optimal processing.

### 3. **Generate Embeddings**
   - Convert the text chunks into dense vector embeddings using Hugging Face models like `sentence-transformers/all-MiniLM-L6-v2`.

### 4. **Store in Milvus**
   - Save the embeddings into a Milvus vector database for efficient retrieval.

### 5. **Build RAG Pipeline**
   - Retrieve relevant chunks using a retriever connected to the vector database.
   - Use a language model to answer questions based on the retrieved context.

### 6. **Interactive Gradio App**
   - Enable real-time interaction by uploading documents and querying the model using a web-based Gradio interface.

---

## Detailed Steps

### Step 1: Document Loading
- **Objective**: Load files into the system and convert them into processable text.
- **Code Snippet**:
  ```python
  loader = DoclingPDFLoader(file_path=FILE_PATH)
  docs = loader.load()
  ```
- **Key Points**:
  - Supports both single and multiple file loading.
  - Handles PDFs and other document formats.
  - Converts files to plain text, markdown, JSON, or YAML.

---

### Step 2: Text Splitting
- **Objective**: Split documents into smaller chunks for efficient processing.
- **Code Snippet**:
  ```python
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  ```
- **Key Points**:
  - Chunk size determines the maximum length of a text chunk.
  - Chunk overlap ensures context continuity between chunks.

---

### Step 3: Generate Embeddings
- **Objective**: Represent text chunks as dense vectors for similarity-based retrieval.
- **Code Snippet**:
  ```python
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  ```
- **Key Points**:
  - Uses pre-trained models from Hugging Face for embeddings.
  - Allows model customization for specific use cases.

---

### Step 4: Store in Milvus
- **Objective**: Save embeddings into a vector database for scalable retrieval.
- **Code Snippet**:
  ```python
  vectorstore = Milvus.from_documents(
      splits,
      embeddings,
      connection_args={"uri": MILVUS_URI},
      drop_old=True,
  )
  ```
- **Key Points**:
  - Milvus efficiently handles large-scale vector storage.
  - Ensures quick retrieval of similar embeddings.

---

### Step 5: Build RAG Pipeline
- **Objective**: Retrieve relevant document chunks and generate context-aware answers.
- **Code Snippet**:
  ```python
  retriever = vectorstore.as_retriever()
  prompt = PromptTemplate.from_template(
      "Context information is below.\n---------------------\n{context}\n---------------------\nQuery: {question}\nAnswer:\n"
  )
  rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
  ```
- **Key Points**:
  - Combines a retriever with a language model.
  - Uses a prompt template to structure queries and retrieved context.

---

### Step 6: Interactive Gradio App
- **Objective**: Provide a user-friendly interface for real-time interaction.
- **Code Snippet**:
  ```python
  demo.launch()
  ```
- **Key Points**:
  - Upload multiple files (PDF, text, images, etc.).
  - Ask questions and view model-generated answers in real time.

---

## Example Queries

1. **Check Document Details**:
   ```python
   rag_chain.invoke("Does Docling implement a linear pipeline of operations?")
   ```

2. **Ask Specific Questions**:
   ```python
   rag_chain.invoke("How many pages were human-annotated for DocLayNet?")
   ```

---

## Gradio Interface

Run the Gradio interface for an interactive experience:

```python
demo.launch()
```

### Features:
- File upload for multiple formats.
- Query input box for questions.
- Real-time responses displayed in the output.

---

## File Structure

```
docling_RAG_langchain/
├── docling_RAG_langchain_colab.ipynb   # Main notebook implementation
├── requirements.txt                   # List of required Python packages
├── LICENSE                            # License file
└── README.md                          # Documentation file
```

---

## References

1. [Docling Documentation](https://ds4sd.github.io/docling/examples/rag_langchain/)
2. [LangChain Documentation](https://python.langchain.com/en/latest/index.html)
3. [Hugging Face Models](https://huggingface.co/models)
4. [Milvus Documentation](https://milvus.io/docs)

---

## Author

- **Partha Pratim Ray**
  - [GitHub](https://github.com/ParthaPRay)
  

Feel free to open issues or submit pull requests to contribute to the project!
