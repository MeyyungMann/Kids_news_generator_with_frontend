# Kids News Generator

A full-stack application that generates kid-friendly news articles. The application uses AI and powered by Retrieval-Augmented Generation (RAG) to combines real-time news retrieval with advanced language models to create engaging, accurate, and contextually relevant content for young readers.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Building for Production](#building-for-production)
- [Technologies Used](#technologies-used)

## Features
- 🤖 AI-powered news content adaptation using transformers
- 🔍 RAG-based content generation with FAISS vector search
- 🖼️ CLIP-powered image generation and understanding
- 📱 Responsive design for all devices
- 📊 Interactive news dashboard
- 🎨 Kid-friendly interface
- 📝 Customizable content filters
- 🌐 Real-time news updates
- 📈 Age-appropriate content adaptation
- 💬 Interactive feedback system

## Project Structure
```
kids-news-generator/
├── frontend/           # React TypeScript application
│   ├── src/           # Source code
│   │   ├── components/  # React components
│   │   ├── pages/      # Page components
│   │   └── App.jsx     # Main application
│   └── package.json   # Frontend dependencies
├── backend/           # Python FastAPI server
│   ├── src/          # Source code
│   │   ├── app.py    # FastAPI application
│   │   ├── ml_pipeline.py  # ML processing
│   │   ├── rag_system.py   # RAG implementation
│   │   ├── clip_handler.py # CLIP integration
│   │   └── config.py      # Configuration
│   ├── data/         # Data storage
│   ├── logs/         # Application logs
│   ├── results/      # Generated results
│   └── requirements.txt  # Backend dependencies
└── venv/             # Python virtual environment
```

## Prerequisites

### Frontend
- Node.js (v14 or higher)
- npm (v6 or higher)

### Backend
- Python 3.8 or higher
- pip (Python package manager)
- CUDA-compatible GPU (recommended for ML features)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/kids-news-generator.git
cd kids-news-generator
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Development Mode

1. Start the backend server:
```bash
cd backend/src
source ../../venv/Scripts/activate  # On Windows
uvicorn app:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173 (Vite development server)
- Backend API: http://localhost:8000 (FastAPI server)

### Example API Usage
```bash
# Get latest news articles
curl http://localhost:8000/api/news

# Get article by ID
curl http://localhost:8000/api/news/123

# Generate kid-friendly article
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "space exploration", "age_group": 8, "category": "Science"}'
```

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
FASTAPI_APP=src.app:app
ENVIRONMENT=development
DATABASE_URL=your_database_url
NEWS_API_KEY=your_news_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

### Frontend Configuration

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
VITE_ENV=development
```

## Development

### Frontend
- The main application code is in `frontend/src/`
- Styles are managed through Tailwind CSS
- API calls are made to the backend server using Axios
- Uses Vite as the build tool and development server

### Backend
- The main server code is in `backend/src/app.py`
- API endpoints are defined using FastAPI
- RAG system implemented in `rag_system.py`
- ML pipeline in `ml_pipeline.py`
- CLIP integration in `clip_handler.py`
- Environment variables are managed through `.env` file

## Building for Production

### Frontend
```bash
cd frontend
npm run build
```

The build artifacts will be stored in the `frontend/dist/` directory.

### Backend
The backend is ready for production deployment. Make sure to:
1. Set up proper environment variables
2. Configure your production server (e.g., Uvicorn with multiple workers)
3. Set up proper security measures

## Technologies Used

### Frontend
- **Core Framework**
  - React 18.2.0 - Modern UI library
  - TypeScript 5.2.2 - Type-safe JavaScript
  - Vite 5.1.0 - Build tool and dev server

- **Styling & UI**
  - Tailwind CSS 3.4.1 - Utility-first CSS framework
  - Headless UI 1.7.18 - Unstyled UI components
  - Heroicons 2.1.1 - Icon set

- **State Management & Data Fetching**
  - React Router 6.22.1 - Routing
  - Axios 1.6.7 - HTTP client
  - React Hooks - State management

- **Development Tools**
  - ESLint 8.56.0 - Code linting
  - TypeScript ESLint - TypeScript linting
  - Vite - Development server

### Backend
- **Core Framework**
  - FastAPI 0.104.1 - Modern web framework
  - Uvicorn 0.24.0 - ASGI server
  - Pydantic 2.4.2 - Data validation

- **Machine Learning & RAG**
  - PyTorch 2.2.0 - Deep learning framework
  - Transformers 4.36.0 - NLP models
  - FAISS 1.7.4 - Vector similarity search
  - Sentence Transformers 2.2.2 - Text embeddings
  - CLIP - Image-text understanding

- **Data Processing**
  - NumPy 1.26.4 - Numerical computing
  - Scikit-learn 1.4.0 - Machine learning
  - BeautifulSoup4 - Web scraping

- **Development Tools**
  - Python-dotenv - Environment variables
  - Requests - HTTP client
  - FastAPI CORS - Cross-origin resource sharing