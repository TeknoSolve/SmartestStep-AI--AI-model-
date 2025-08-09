
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
import pdfplumber
from docx import Document
import re
import spacy
import groq
import tempfile
import uuid
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load SpaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Set your Groq API key
GROQ_API_KEY = "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"
client = groq.Client(api_key=GROQ_API_KEY)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

# ========== UTILITY FUNCTIONS ==========

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text if text.strip() else None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text.strip() else None
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

# ✅ Extract user name from resume text
def extract_user_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip().replace(" ", "_")
    return f"user_{uuid.uuid4().hex[:6]}"  # fallback if name not found

# ✅ Store resume text to a file named by user's name
def store_resume_text_locally(text):
    user_name = extract_user_name(text)
    filename = f"{user_name}.txt"
    folder = "resumes"
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Resume saved at: {full_path}")
    return full_path

# ========== RESUME VALIDATION AND ANALYSIS ==========

def is_valid_resume(text):
    if text is None or len(text.split()) < 150:
        return False

    doc = nlp(text)

    section_patterns = re.compile(
        r'(work\s*(experience|history)|employment|education|skills|technical\s*skills|projects?|'
        r'certifications?|awards|achievements|summary|objective|profile|responsibilities)',
        re.IGNORECASE
    )
    found_sections = len(set(section_patterns.findall(text.lower())))

    required_entities = {
        "PERSON", "ORG", "GPE", "DATE", "LOC", "FAC", "NORP", "LANGUAGE", "WORK_OF_ART", "PRODUCT"
    }
    entity_counts = sum(1 for ent in doc.ents if ent.label_ in required_entities)

    has_email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    has_phone = re.search(r"\+?\d[\d\s\-]{7,}", text)
    has_linkedin = re.search(r"linkedin\.com/in/\w+", text)
    has_contact_info = has_email or has_phone or has_linkedin

    skills_keywords = ["python", "sql", "java", "html", "css", "machine learning", "aws", "c++", "git"]
    has_skills_keywords = any(skill in text.lower() for skill in skills_keywords)

    return all([
        found_sections >= 3,
        entity_counts >= 3,
        has_contact_info,
        has_skills_keywords
    ])

def generate_resume_analysis(resume_text):
    prompt = f"""You are a Resume Evaluation AI. Evaluate the given resume based on these criteria:

1. Clarity and organization  
2. Grammar and spelling  
3. Formatting consistency  
4. Relevant skills and experience  
5. Overall professionalism  
6. Use of action verbs & achievements  
7. Relevance to the Job/Industry  
8. Contact Information & Professional Summary  

*Backend Processing:*  
- Assign each criterion a score from 0 to 10  
- Calculate the overall percentage score (0–100%)  
- Show only the final percentage score at the top (e.g., "Score: 78%")  
- Do not show the individual scores  

*User Output:*  
After the score, provide each improvement area in a *structured and easy-to-follow* way:  
1. *Identify the specific issue* (e.g., "Unclear headings")  
2. *Explain briefly why it matters* (why employers care)  
3. *Suggest simple, step-by-step fixes* (as if explaining to a 1st-year student)  

⚠ *Important:*  
- Always output the final score like this: **Score: <score>%**

{resume_text}
"""

    try:
        print("🔍 [Groq] Analyzing resume...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()

        return {
            "passed": result.startswith("✅"),
            "analysis": result
        }
    except Exception as e:
        print("❌ Resume Analysis Error:", str(e))
        import traceback
        traceback.print_exc()
        return {
            "passed": False,
            "analysis": "❌ Internal error while analyzing the resume."
        }

# ========== FLASK ROUTES ==========

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_resume_api():
    resume_text = None

    if "resume_file" in request.files and request.files["resume_file"].filename != "":
        file = request.files["resume_file"]
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in [".pdf", ".doc", ".docx"]:
            return jsonify({"error": "Unsupported file format. Please upload PDF or DOCX."}), 400

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                file.save(tmp.name)
                if ext == ".pdf":
                    resume_text = extract_text_from_pdf(tmp.name)
                else:
                    resume_text = extract_text_from_docx(tmp.name)
            os.unlink(tmp.name)
        except Exception as e:
            print("❌ File processing error:", str(e))
            return jsonify({"error": "Error reading the uploaded file."}), 500

    elif request.form.get("resume_text", "").strip():
        resume_text = request.form.get("resume_text", "").strip()

    else:
        return jsonify({"error": "Please upload a resume file or paste resume text."}), 400

    if not resume_text:
        return jsonify({"error": "Failed to extract text from the provided resume."}), 400

    if not is_valid_resume(resume_text):
        return jsonify({"error": "The resume does not appear valid."}), 400

    # ✅ Store resume text to .txt file using user name
    stored_path = store_resume_text_locally(resume_text)
    print(f"✅ Resume saved at: {stored_path}")

    analysis_result = generate_resume_analysis(resume_text)

    return jsonify({
        "analysis": analysis_result["analysis"],
    })

if __name__ == "__main__":
    app.run(debug=True)
    
