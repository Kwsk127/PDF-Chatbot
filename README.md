# PDF-Chatbot

This project is a PDF-based chatbot powered by the LangChain framework. It allows users to upload multiple PDFs and ask questions related to the content of the PDFs. The chatbot uses a combination of a HuggingFace language model, a FAISS vector store for fast document retrieval, and Groq's chat model to generate answers.

## Features

- **PDF Upload and Processing:** Users can upload one or more PDFs, which are processed and chunked into manageable text sections.
- **Question Answering:** The chatbot retrieves relevant information from the processed PDFs and answers user queries.
- **Custom Prompting:** A custom prompt template ensures that the responses are directly based on the content of the PDFs.
- **Chat History:** A session-based chat history is maintained, enabling users to track the conversation.
- **Vector Store for Fast Retrieval:** FAISS is used as a vector store for efficient search and retrieval of relevant document sections.
- **Groq Integration:** Leverages the ChatGroq model for generating responses to user queries.

## Requirements

- Python 3.x
- Streamlit
- LangChain
- HuggingFace
- FAISS
- Groq API
- dotenv (for loading environment variables)
- PyPDF2

You can install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

## How It Works

1. **Upload PDFs:** Users can upload one or more PDF files via the Streamlit interface.
   
2. **PDF Text Extraction:** The text from the uploaded PDFs is extracted and chunked into smaller sections for efficient processing.

3. **Question Answering:** Users can ask questions related to the content of the PDFs, and the chatbot will retrieve the most relevant document sections and generate an answer using the Groq model.

4. **Chat Interface:** All interactions are stored in a session-based chat history, allowing users to interact with the chatbot in a conversational manner.

## License

This project is licensed under the MIT License.
