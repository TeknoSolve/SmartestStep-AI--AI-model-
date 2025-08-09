# TeknoSolve - AI Interview System

An AI-driven platform that evaluates resumes in real time, scores them, and provides clear, actionable feedback. Designed to help job seekers improve their chances, it delivers instant, ATS-friendly insights and recommendations, empowering users to refine their profiles for maximum visibility and impact in competitive hiring processes.

## Features
- 🎙️ Real-time speech recognition
- 🤖 AI-powered question generation
- 📝 Automatic response evaluation
- 📊 Detailed interview transcripts
- 🎯 Comprehensive final reports

## Tech Stack
- Python 3.10+
- Flask/CORS
- Groq LLM
- Deepgram Speech Recognition
- SpaCy NLP
- LangChain

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
# Create .env file with:
DEEPGRAM_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

3. Initialize directories:
```bash
mkdir resumes transcripts
```

## Usage
Run the interview system:
```bash
python low.py
```

## Project Structure
```
TeknoSolve/
├── app.py          # Flask web application
├── low.py          # Core interview logic
├── static/         # Static assets
├── templates/      # HTML templates
├── resumes/        # Resume storage
└── transcripts/    # Interview transcripts
```

## License
MIT License# tekno-solve
