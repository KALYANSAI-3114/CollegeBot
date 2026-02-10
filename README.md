# CollegeBot 🎓

A smart chatbot built to answer questions about your college using RAG (Retrieval Augmented Generation) technology.

## Overview

CollegeBot is an AI-powered chatbot that helps students, faculty, and visitors get information about your college. It uses web-based data loading to gather college information and provides accurate, context-aware responses using the Sarvam AI API.

## Features

- 💬 Interactive chat interface for college-related queries
- 🌐 Web-based data loading from college website
- 🧠 RAG (Retrieval Augmented Generation) for accurate responses
- 🔍 Vector-based semantic search using ChromaDB
- ⚡ Fast and efficient information retrieval
- 🎨 User-friendly Streamlit interface

## Tech Stack

- **Frontend**: HTML,CSS,JS
- **Backend**:FastAPI
- **LLM API**: Sarvam AI API
- **Vector Database**: ChromaDB
- **Data Loading**: Web-based loader
- **Language**: Python

## Prerequisites

Before running this project, make sure you have:

- Python 3.8 or higher
- Sarvam AI API key
- Internet connection for web scraping

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/KALYANSAI-3114/CollegeBot.git
cd CollegeBot
cd backend
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
SARVAM_API_KEY=your_sarvam_api_key_here
```

## Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Access the chatbot**

Open your browser and navigate to `http://localhost:8501`

3. **Start chatting**

Ask questions about your college such as:
- "What courses are offered?"
- "What are the admission requirements?"
- "Tell me about campus facilities"
- "What is the fee structure?"

## Configuration

### Data Sources

Update the web URLs in your data loader configuration to point to your college website pages.


## How It Works

1. **Data Loading**: The web-based loader scrapes information from your college website
2. **Chunking**: Text is split into manageable chunks for processing
3. **Embedding**: Chunks are converted to vector embeddings
4. **Storage**: Embeddings are stored in ChromaDB for fast retrieval
5. **Query Processing**: User questions are converted to embeddings
6. **Retrieval**: Relevant documents are retrieved using semantic search
7. **Generation**: Sarvam AI generates contextual responses using retrieved information

## Dependencies

```txt
#Force CPU-only PyTorch (Python 3.13 compatible)
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.10.0+cpu

# Sentence embeddings
sentence-transformers
pysqlite3-binary==0.5.3

# LangChain stack
langchain
langchain-community
langchain-core
langchain-openai
langchain-huggingface
chromadb

# Parsing
beautifulsoup4
lxml

fastapi
uvicorn
pydantic
```


**Live Demo**: https://collegebot-3114.streamlit.app/

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SARVAM_API_KEY` | Your Sarvam AI API key for LLM access |

## Troubleshooting

### Common Issues

**Issue**: ChromaDB persistence error
- **Solution**: Ensure the data directory has proper write permissions

**Issue**: API rate limiting
- **Solution**: Implement rate limiting or upgrade your Sarvam API plan

**Issue**: Slow response times
- **Solution**: Reduce chunk size or limit the number of retrieved documents

## Future Enhancements

- [ ] Add support for multiple languages
- [ ] Include image-based queries
- [ ] Add chat history persistence
- [ ] Implement user feedback mechanism
- [ ] Add analytics dashboard
- [ ] Support for document uploads

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- [Sarvam AI](https://www.sarvam.ai/) for providing the LLM API
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Streamlit](https://streamlit.io/) for the deployment platform
- [LangChain](https://www.langchain.com/) for the RAG framework

## Contact

Your Name - [kalyansai0909@gmail.com]

Project Link: [https://github.com/KALYANSAI-3114/CollegeBot/](https://github.com/KALYANSAI-3114/CollegeBot/)

---
