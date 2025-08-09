<<<<<<< HEAD
import asyncio
import os
import time 
import spacy
import uuid
import re

from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()
# ⭐️ Update: AI Question extraction from transcripts is now straightforward
def get_all_previous_questions(user_id, transcript_folder="transcripts"):
    question_set = set()
    if not os.path.exists(transcript_folder):
        return question_set
    for filename in os.listdir(transcript_folder):
        if filename.startswith(f"interview_{user_id}_") and filename.endswith(".txt"):
            with open(os.path.join(transcript_folder, filename), "r", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("🔹AI Question:"):
                        question = line.split(":",1)[1].strip()
                        if question:
                            question_set.add(question)
    return question_set

# ⭐️ NEW: Extract exact technical question (last sentence ending with ?)
def extract_actual_question(ai_response):
    questions = re.findall(r"([^?]+\?)", ai_response, flags=re.DOTALL)
    if questions:
        return questions[-1].strip()
    return ai_response.strip()

def extract_skills(resume_text: str) -> List[str]:
    """
    Extract technical skills from resume text.
    Returns a list of identified technical skills.
    """
    # Common technical skills to look for (case-insensitive)
    technical_skills = {
        # Programming Languages
        'python', 'java', 'javascript', 'c++', 'c#', 'c', 'php', 'ruby', 'go', 'rust',
        'swift', 'kotlin', 'typescript', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash',
        'assembly', 'cobol', 'fortran', 'pascal', 'delphi', 'vb.net', 'visual basic',
        
        # Web Technologies
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'express', 'django',
        'flask', 'spring', 'bootstrap', 'jquery', 'sass', 'less', 'webpack', 'babel',
        'next.js', 'nuxt.js', 'gatsby', 'svelte', 'ember', 'backbone',
        
        # Databases
        'mysql', 'postgresql', 'postgres', 'mongodb', 'sqlite', 'oracle', 'redis', 'cassandra',
        'dynamodb', 'firebase', 'mariadb', 'neo4j', 'couchdb', 'elasticsearch', 'solr',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'gitlab', 'bitbucket', 'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'nginx',
        'apache', 'tomcat', 'heroku', 'netlify', 'vercel',
        
        # Data Science & ML
        'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras',
        'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'tableau', 'power bi',
        'opencv', 'nltk', 'spacy', 'statsmodels', 'scipy',
        
        # Mobile Development
        'android', 'ios', 'flutter', 'react native', 'xamarin', 'cordova', 'phonegap',
        'ionic', 'unity', 'kotlin', 'objective-c',
        
        # Other Tools & Technologies
        'linux', 'ubuntu', 'windows', 'macos', 'jira', 'confluence', 'slack', 'postman',
        'vs code', 'visual studio', 'intellij', 'eclipse', 'vim', 'emacs', 'sublime',
        'figma', 'sketch', 'photoshop', 'illustrator', 'blender', 'unity3d',
        'rabbitmq', 'kafka', 'redis', 'memcached', 'graphql', 'rest', 'soap',
        'json', 'xml', 'yaml', 'csv', 'excel', 'word', 'powerpoint'
    }
    
    # Convert resume to lowercase for matching
    resume_lower = resume_text.lower()
    
    # Find skills mentioned in the resume
    found_skills = []
    for skill in technical_skills:
        # Special handling for single-letter skills like "C"
        if skill.lower() == 'c':
            # Look for "C" as standalone or followed by non-alphanumeric (but not "+")
            pattern = r'\bC\b(?!\+)'
            if re.search(pattern, resume_text):  # Case sensitive for "C"
                found_skills.append('C')
        else:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, resume_lower):
                found_skills.append(skill.title())  # Capitalize for display
    
    # Look for skills in common sections
    skills_sections = re.findall(r'(?:skills?|technologies?|tools?)[:\-\s]*([^\n\r]*(?:\n(?!\s*\n)[^\n\r]*)*)', 
                                resume_lower, re.IGNORECASE | re.MULTILINE)
    
    for section in skills_sections:
        # Extract individual skills from comma-separated or bullet-pointed lists
        items = re.findall(r'[•\-\*]?\s*([a-zA-Z0-9\+\#\.]+(?:\s+[a-zA-Z0-9\+\#\.]+)*)', section)
        for item in items:
            item = item.strip()
            if len(item) > 1 and item.lower() in [s.lower() for s in technical_skills]:
                if item.title() not in found_skills:
                    found_skills.append(item.title())
    
    return list(set(found_skills))  # Remove duplicates


def extract_projects(resume_text: str) -> List[Dict[str, str]]:
    """
    Extract project information from resume text.
    Returns a list of dictionaries with project details.
    """
    projects = []
    
    # Common project section headers
    project_patterns = [
        r'(?:projects?|personal projects?|academic projects?|work projects?)[:\-\s]*\n',
        r'(?:portfolio|github projects?)[:\-\s]*\n'
    ]
    
    # Find project sections
    for pattern in project_patterns:
        matches = re.finditer(pattern, resume_text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            # Get text after the project header
            start_pos = match.end()
            
            # Find the next major section or end of text
            next_section = re.search(r'\n(?:[A-Z][A-Z\s]{3,}|EXPERIENCE|EDUCATION|SKILLS|CERTIFICATIONS)', 
                                   resume_text[start_pos:], re.IGNORECASE)
            
            if next_section:
                project_text = resume_text[start_pos:start_pos + next_section.start()]
            else:
                project_text = resume_text[start_pos:]
            
            # Extract individual projects
            project_entries = extract_individual_projects(project_text)
            projects.extend(project_entries)
    
    # Also look for project-like entries throughout the resume
    additional_projects = find_standalone_projects(resume_text)
    projects.extend(additional_projects)
    
    # Remove duplicates based on project title
    unique_projects = []
    seen_titles = set()
    
    for project in projects:
        title_lower = project['title'].lower().strip()
        if title_lower not in seen_titles and len(title_lower) > 2:
            seen_titles.add(title_lower)
            unique_projects.append(project)
    
    return unique_projects[:10]  # Limit to top 10 projects


def extract_individual_projects(project_text: str) -> List[Dict[str, str]]:
    """Helper function to extract individual projects from a project section."""
    projects = []
    
    # Split by common project separators
    project_blocks = re.split(r'\n(?=\s*[•\-\*]|\s*\d+\.|\s*[A-Z][a-zA-Z\s]+:)', project_text)
    
    for block in project_blocks:
        if len(block.strip()) < 10:  # Skip very short blocks
            continue
            
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
        
        # First line is likely the project title
        title_line = lines[0]
        
        # Clean up title (remove bullets, numbers, etc.)
        title = re.sub(r'^[•\-\*\d\.\s]*', '', title_line).strip()
        title = re.sub(r'[:\-\|].*$', '', title).strip()  # Remove description after colon/dash
        
        if len(title) < 3:
            continue
        
        # Get description from remaining lines
        description_lines = lines[1:] if len(lines) > 1 else []
        description = ' '.join(description_lines)[:200]  # Limit description length
        
        # Extract technologies mentioned in this project
        technologies = extract_project_technologies(block)
        
        projects.append({
            'title': title,
            'description': description,
            'technologies': ', '.join(technologies) if technologies else ''
        })
    
    return projects


def find_standalone_projects(resume_text: str) -> List[Dict[str, str]]:
    """Find project-like entries that might not be in a dedicated project section."""
    projects = []
    
    # Look for lines that might be project titles
    lines = resume_text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip very short or very long lines
        if len(line) < 5 or len(line) > 100:
            continue
        
        # Look for patterns that suggest a project title
        project_indicators = [
            r'\b(?:developed|built|created|implemented|designed)\b.*(?:application|app|website|system|tool)',
            r'\b(?:e-commerce|portfolio|dashboard|chatbot|game)\b',
            r'\b(?:web|mobile|desktop)\s+(?:app|application)\b'
        ]
        
        for pattern in project_indicators:
            if re.search(pattern, line, re.IGNORECASE):
                # Extract a clean title
                title = re.sub(r'^[•\-\*\d\.\s]*', '', line).strip()
                
                # Get some context from surrounding lines
                context_lines = []
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    if j != i and lines[j].strip():
                        context_lines.append(lines[j].strip())
                
                description = ' '.join(context_lines)[:150]
                
                projects.append({
                    'title': title,
                    'description': description,
                    'technologies': ''
                })
                break
    
    return projects


def extract_project_technologies(project_text: str) -> List[str]:
    """Extract technologies mentioned in a specific project description."""
    # Common technical terms to look for in project descriptions
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'html', 'css',
        'mysql', 'mongodb', 'postgresql', 'django', 'flask', 'spring',
        'aws', 'azure', 'docker', 'git', 'github', 'api', 'rest'
    ]
    
    found_tech = []
    project_lower = project_text.lower()
    
    for tech in tech_keywords:
        if re.search(r'\b' + re.escape(tech) + r'\b', project_lower):
            found_tech.append(tech.title())
    
    return found_tech



# ✅ Final Evaluation Prompt Generator
def generate_final_evaluation_prompt(full_text):
    return  f"""
You are SmartestStep AI — a highly experienced communication and knowledge assessment mentor.

You've reviewed thousands of interview transcripts. Your job is to evaluate how well the candidate performed in terms of communication and subject knowledge. Do not penalize based on accent — focus on clarity, structure, tone, and actual content accuracy.

---

### 🧠 Communication Evaluation Criteria:

1. **Tone**
   - Did the candidate sound calm, natural, and professionally engaged?

2. **Clarity & Structure**
   - Were their ideas well-organized and easy to understand?

3. **Confidence**
   - Did they speak with self-assurance and minimal hesitation?

For each, score from 0–10 and label:
- Low: 0–3
- Medium: 4–7
- High: 8–10

---

### 🧪 Knowledge Evaluation:

- Review the transcript and identify the number of questions attempted by the candidate.
- For each response, classify it as:
  - ✅ **Correct**
  - ⚠️ **Partially Correct**
  - ❌ **Incorrect or Unclear**

- Use the following formula:
  - **Answer Accuracy %** = (Correct + 0.5 × Partially Correct) / Total Questions × 100
  - **Answer Accuracy Score (0–10)** = Answer Accuracy % / 10

---

### 🧮 Final Interview Score:

- **Communication Score (0–10)** = Average of Tone, Clarity, Confidence
- **Final Interview Score (0–10)** = Average of Communication Score and Answer Accuracy Score

---

### ✅ Pass/Fail Logic:

- If **Final Interview Score ≥ 9.0 (90%)**:
  ✅ “You have passed this round and may proceed to the next stage.”
- Else:
  ❌ “You did not pass. Please review the feedback and try again.”

---

### 🗣 Input:
Full Interview Transcript (Only Candidate's Spoken Responses):
\"\"\"{full_text}\"\"\"

---

### 📤 Output Format:

**Tone Score**: [0–10], Level: [Low / Medium / High]  
**Clarity & Structure Score**: [0–10], Level: [Low / Medium / High]  
**Confidence Score**: [0–10], Level: [Low / Medium / High]  
**Overall Communication Score**: [0–10]

📚 **Knowledge Evaluation**:  
- Total Questions: [X]  
- Correct Answers: [Y]  
- Partially Correct Answers: [Z]  
- Incorrect Answers: [W]  
- Answer Accuracy Score: [0–10]

🎯 **Final Interview Score**: [0–10]  
✅/❌ **Interview Status**: [Pass or Fail Message]

---

📝 **Feedback Summary**:  
- [Brief summary on strengths and weaknesses]

🔧 **Top 3 Suggestions for Improvement**:  
1. [Tip 1]  
2. [Tip 2]  
3. [Tip 3]
"""




# ✅ LLM processor
# 💡 CHANGED: LLM processor now supports get/set of excluded questions
class LanguageModelProcessor:
    def __init__(
        self,
        resume_text,
        stage="Basic",
        candidate_type="unknown",
        skills_list=None,
        projects_list=None,
        excluded_questions=None  # 💡 NEW
    ):
        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Store excluded questions (from transcript history)
        self.excluded_questions = excluded_questions or set()  # 💡 NEW

        system_prompt = generate_stage_prompt(
            stage,
            resume_text,
            candidate_type,
            skills_list,
            projects_list,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}"),
        ])

        self.conversation = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        start = time.time()
        response = self.conversation.invoke({"text": text})
        self.memory.chat_memory.add_ai_message(response["text"])
        print(f"LLM ({int((time.time() - start) * 1000)}ms): {response['text']}")
        return response["text"]

    def evaluate_final_report(self, full_text):
        prompt = generate_final_evaluation_prompt(full_text)
        start = time.time()
        response = self.conversation.invoke({"text": prompt})
        print(f"Final Evaluation LLM ({int((time.time() - start) * 1000)}ms):\n{response['text']}")
        return response["text"]

    def evaluate_answer(self, question, answer):
        # [UNCHANGED...]
        evaluation_prompt = f"""
                            You are an interview evaluator.
                            Question: {question}
                            Candidate Answer: {answer}

                            1. ✅ Give a correct answer.
                            2. 🧠 Then, evaluate the candidate's answer and classify it with **exactly one** of these three categories (use these exact phrases):
                                - Correct
                                - Partially correct
                                - Incorrect
                            Do NOT use any other words, combinations, or synonyms.
                            3. Provide a short reason for your evaluation.
                            Format:
                            Ideal Answer: ...
                            Answer Check: ...
        """
        response = self.llm.invoke(evaluation_prompt).content.strip()

        # Split response cleanly
        lines = response.split("\n")
        ideal_answer = ""
        correctness = ""

        for line in lines:
            if line.lower().startswith("ideal answer"):
                ideal_answer = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer check"):
                correctness = line.split(":", 1)[1].strip()

        return ideal_answer, correctness
# Updated generate_stage_prompt function with skill and project extraction
def generate_stage_prompt(stage, resume_text, candidate_type="unknown", skills_list=None, projects_list=None):
    # [UNCHANGED: ...as in your code, except ADD instructions to not repeat from previous sessions]
    if skills_list is None:
        skills_list = extract_skills(resume_text)
    if projects_list is None:
        projects_list = extract_projects(resume_text)

    skills_text = ", ".join(skills_list[:8]) if skills_list else "general technical concepts"
    projects_text = ""
    if projects_list:
        projects_text = "\n   - Projects to reference:\n"
        for project in projects_list[:5]:  # Limit to top 5 projects
            projects_text += f"     • {project['title']}"
            if project.get('technologies'):
                projects_text += f" (using {project['technologies']})"
            projects_text += "\n"
    
    return f"""
You are a friendly, professional virtual interviewer conducting the **{stage} Technical Stage** of a technical interview.

🎯 OBJECTIVE:
- Evaluate the candidate's **advanced technical knowledge and practical application**.
- Detect whether the candidate is a **student** or a **working professional**, and adjust tone/wording slightly — but do NOT change question complexity.
- Ask only **advanced-level questions**, deeply tied to the candidate’s listed skills and projects. Avoid theoretical trivia.

📋 SESSION INFO:
- Candidate Type: {candidate_type}
- Key Skills to Focus On: {skills_text}
- Resume Context: {resume_text[:500]}...{projects_text}

---

🧠 QUESTIONING RULES:
1. Ask **exactly one question per turn**.
2. Ask a total of **12–15 advanced-level technical questions**, then conclude the interview.

3. Only ask **advanced questions**, such as:
   - Scenario-based questions (e.g., "How would you handle X if Y occurred?")
   - Architectural or optimization tradeoffs
   - Advanced debugging, design, or deployment issues
   - Questions that explore reasoning, critical thinking, and experience

4. Rotate through **2–3 key technical skills** from these extracted skills: {skills_text}
   - Ask **2–4 advanced questions per skill**.
   - Do NOT ask about any topic, library, or tool not listed in the skills section.

5. Deeply leverage listed projects:
   - Ask **real-world**, **experience-based** questions like:
     - "In project X, how did you address [challenge]?"
     - "How would you scale/improve project Y today?"
     - "What trade-offs did you make when designing feature Z?"
   - Ask 2–3 project-specific questions spaced out across the session.

6. Customize wording based on candidate type:
   👨‍🎓 *Student* → focus on deep understanding and strong reasoning
   👩‍💼 *Professional* → ask workplace-scenario questions or cross-functional impact
   *(Both should face advanced technical difficulty — not behavioral questions)*

---

⚠️ ADVANCED QUESTION TONE REQUIREMENTS:
- Do NOT ask definitions or trivia.
- Do NOT ask basic-level comparisons or conceptual theory.
- Do NOT explain questions — challenge the candidate to reason.
- Do NOT repeat topics/questions from Basic or Intermediate rounds.

---

💬 RESPONSE RULES:

1. For each turn, your reply MUST ONLY consist of the *next technical question* to ask.
2. Do NOT output "Noted.", "Thank you.", "Let's move on", "Let's skip", "That's okay", or any similar transition or acknowledgment phrases.
3. Do NOT recap the previous question or answer.
4. Do NOT provide feedback, encouragement, summaries, or any polite filler.
5. Your output should always be a single, **direct**, **technical**, **advanced-level** question ending with a "?" — nothing else.

---

🚫 ABSOLUTELY DO NOT:
- ❌ Explain or define correct answers
- ❌ Give feedback like "That's right" or "Actually…"
- ❌ Teach or correct any concepts
- ❌ Repeat questions (even with slightly different wording) — across this session or previous ones
- ❌ Ask compound/multi-part questions
- ❌ Use basic or intermediate concepts (e.g., "What is a variable?")
- ❌ Ask HR, behavioral, or soft-skill questions
- ❌ Include transition phrases (e.g., "Let's move on", "That's okay")

---

🛑 RESPONSE HANDLING RULES:

- For a candidate answer:  
  - ✅ On most turns, respond ONLY with the next technical question.
- If the candidate says "I don't know", "I'm not sure", or "Can we skip?":
  - ✅ Respond ONLY with the next technical question (no filler).
- If silent for 10 seconds:
  - ✅ Respond: "Are you still there? Please take your time and answer when you're ready."
  - If still silent after another 10 seconds, directly ask the next technical question (no filler).
- If the candidate asks a question (e.g., "What do you mean?" or "Can you explain?"):
  - ✅ Respond: "Let's focus on the current question. Please answer first — we'll clarify afterward."

---

📌 FEEDBACK POLICY:
- Do NOT explain, correct, or guide the candidate.
- Feedback will be provided only after the full session ends.
- If you begin explaining, immediately stop and say: "Noted." Move to the next question.

---

🏁 ENDING INSTRUCTIONS:
After 12–15 questions, end with:
**"We've completed the {stage} Technical Stage. Thank you for your time. Your responses will now be reviewed."**
"""



# ✅ Load latest resume text file
def get_latest_resume_text():
    resume_folder = "resumes"
    if not os.path.exists(resume_folder):
        raise Exception("❌ 'resumes' folder does not exist.")

    txt_files = [f for f in os.listdir(resume_folder) if f.endswith(".txt")]
    if not txt_files:
        raise Exception("❌ No .txt resume found in the 'resumes' folder.")

    txt_files.sort(key=lambda f: os.path.getmtime(os.path.join(resume_folder, f)), reverse=True)
    latest_file = os.path.join(resume_folder, txt_files[0])

    print(f"✅ Latest resume detected: {latest_file}")
    with open(latest_file, "r", encoding="utf-8") as f:
        return f.read().strip()


# ✅ Detect candidate type
def detect_candidate_type(resume_text):
    student_keywords = ["student", "undergraduate", "pursuing", "bachelor", "b.tech", "bsc", "bca", "college", "university", "currently studying"]
    professional_keywords = ["experience", "worked", "developer", "engineer", "software", "company", "organization", "intern", "employment"]

    student_score = sum(kw in resume_text.lower() for kw in student_keywords)
    professional_score = sum(kw in resume_text.lower() for kw in professional_keywords)

    if student_score > professional_score:
        return "student"
    elif professional_score > student_score:
        return "professional"
    elif student_score == professional_score and student_score > 0:
        return "both"
    else:
        return "unknown"



# ✅ Transcript helper
class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return " ".join(self.transcript_parts)

transcript_collector = TranscriptCollector()

# ✅ Corrected get_transcript function
async def get_transcript(callback):
    try:
        transcription_complete = asyncio.Event()

        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        loop = asyncio.get_running_loop()

        async def on_message(connection, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript

            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript().strip()
                if full_sentence:
                    print(f"User: {full_sentence}")
                    await callback(full_sentence)
                    transcript_collector.reset()
                    transcription_complete.set()

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=500,
            smart_format=True,
        )

        await dg_connection.start(options)

        microphone = Microphone(
            lambda data: loop.call_soon_threadsafe(
                lambda: asyncio.create_task(dg_connection.send(data))
            )
        )
        microphone.start()

        await transcription_complete.wait()

        microphone.finish()
        await dg_connection.finish()

    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")

# Set your user_id here for now. Change this when integrating a frontend.
USER_ID = "test_candidate"  # <---- substitute with "candidate_email@example.com" later

class ConversationManager:
    def __init__(self, resume_text, stage="Basic", candidate_type="unknown", user_id="test_candidate"):
        self.full_transcript_list = []
        self.human_responses = []
        self.resume_text = resume_text
        self.user_id = user_id
        print(f"🔑 User ID for this session: {self.user_id}")

        self.transcript_filename = f"transcripts/interview_{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("transcripts", exist_ok=True)

        with open(self.transcript_filename, "w", encoding="utf-8") as f:
            f.write(f"=== INTERVIEW SESSION STARTED ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stage: {stage}\n")
            f.write(f"Candidate Type: {candidate_type}\n")
            f.write("="*50 + "\n\n")

        self.previous_questions = get_all_previous_questions(self.user_id)
        self.llm = LanguageModelProcessor(
            resume_text=resume_text,
            stage=stage,
            candidate_type=candidate_type,
            excluded_questions=self.previous_questions
        )

        self.transcript_queue = asyncio.Queue()
        print(f"🧠 Resume Loaded. Beginning {stage} Stage Interview...")
        print(f"📝 Transcript will be saved to: {self.transcript_filename}\n")
        # The first welcome/question is generated here
        self.llm.process("Hello, I am ready to begin the interview.")

    async def listen_and_respond(self):
        async def handle_full_sentence(full_sentence):
            await self.transcript_queue.put(full_sentence)

        while True:
            # Save current AI question BEFORE listening, so it's always the exact prompt given to the user.
            ai_question_for_transcript = ""
            if self.llm.memory.chat_memory.messages:
                ai_question_for_transcript = self.llm.memory.chat_memory.messages[-1].content.strip()

            print("🎙️ Listening for speech...")
            await get_transcript(handle_full_sentence)

            while not self.transcript_queue.empty():
                transcript = await self.transcript_queue.get()
                if not transcript:
                    print("⚠️ No transcription received. Retrying...")
                    continue

                if "goodbye" in transcript.lower():
                    print("👋 Goodbye detected. Generating final report and ending session.")
                    await self.generate_report_and_exit()
                    return

                self.human_responses.append(transcript)
                
                ai_response = self.llm.process(transcript)
                # We add the question just asked to excluded set for this session
                self.llm.excluded_questions.add(ai_question_for_transcript)
                ideal_answer, correctness = self.llm.evaluate_answer(ai_question_for_transcript, transcript)

                entry = f"""🔹AI Question: {ai_question_for_transcript}
👤 User Answer: {transcript}
✅ Ideal Answer: {ideal_answer}
🧠 Answer Check: {correctness}
{"-"*50}
"""
                if entry not in self.full_transcript_list:
                    self.full_transcript_list.append(entry)
                    try:
                        with open(self.transcript_filename, "a", encoding="utf-8") as f:
                            f.write(entry + "\n")
                    except Exception as e:
                        print(f"❌ Failed to save Q&A: {e}")

    async def generate_report_and_exit(self):
        unique_responses = list(dict.fromkeys(self.human_responses))
        print("🔄 Generating final evaluation report...")
        final_report = self.llm.evaluate_final_report("\n".join(unique_responses))
        
        try:
            with open(self.transcript_filename, "a", encoding="utf-8") as f:
                f.write("\n" + "="*60 + "\n")
                f.write("📊 FINAL EVALUATION REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(final_report)
                f.write(f"\n\nSession completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"✅ Complete interview transcript saved to: {self.transcript_filename}")
            print("\n" + "="*60)
            print("📊 FINAL EVALUATION REPORT")
            print("="*60)
            print(final_report)
        except Exception as e:
            print(f"❌ Failed to save final report: {e}")

        if os.path.exists("final.txt"):
            try:
                os.remove("final.txt")
                print("🧹 Cleaned up temporary final.txt file")
            except Exception as e:
                print(f"⚠️ Could not remove final.txt: {e}")

    async def main(self):
        await self.listen_and_respond()
        if not self.full_transcript_list:
            print("⚠️ No transcript collected. Skipping report.")
            return

   


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    try:
        resume_text = get_latest_resume_text()
        candidate_type = detect_candidate_type(resume_text)
        print(f"🧠 Detected Candidate Type: {candidate_type}")
        user_id = USER_ID
        manager = ConversationManager(resume_text, stage="Basic", candidate_type=candidate_type,user_id=user_id)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(manager.main())
    except Exception as err:
=======
import asyncio
import os
import time 
import spacy
import uuid
import re

from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()
# ⭐️ Update: AI Question extraction from transcripts is now straightforward
def get_all_previous_questions(user_id, transcript_folder="transcripts"):
    question_set = set()
    if not os.path.exists(transcript_folder):
        return question_set
    for filename in os.listdir(transcript_folder):
        if filename.startswith(f"interview_{user_id}_") and filename.endswith(".txt"):
            with open(os.path.join(transcript_folder, filename), "r", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("🔹AI Question:"):
                        question = line.split(":",1)[1].strip()
                        if question:
                            question_set.add(question)
    return question_set

# ⭐️ NEW: Extract exact technical question (last sentence ending with ?)
def extract_actual_question(ai_response):
    questions = re.findall(r"([^?]+\?)", ai_response, flags=re.DOTALL)
    if questions:
        return questions[-1].strip()
    return ai_response.strip()

def extract_skills(resume_text: str) -> List[str]:
    """
    Extract technical skills from resume text.
    Returns a list of identified technical skills.
    """
    # Common technical skills to look for (case-insensitive)
    technical_skills = {
        # Programming Languages
        'python', 'java', 'javascript', 'c++', 'c#', 'c', 'php', 'ruby', 'go', 'rust',
        'swift', 'kotlin', 'typescript', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash',
        'assembly', 'cobol', 'fortran', 'pascal', 'delphi', 'vb.net', 'visual basic',
        
        # Web Technologies
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'express', 'django',
        'flask', 'spring', 'bootstrap', 'jquery', 'sass', 'less', 'webpack', 'babel',
        'next.js', 'nuxt.js', 'gatsby', 'svelte', 'ember', 'backbone',
        
        # Databases
        'mysql', 'postgresql', 'postgres', 'mongodb', 'sqlite', 'oracle', 'redis', 'cassandra',
        'dynamodb', 'firebase', 'mariadb', 'neo4j', 'couchdb', 'elasticsearch', 'solr',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'gitlab', 'bitbucket', 'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'nginx',
        'apache', 'tomcat', 'heroku', 'netlify', 'vercel',
        
        # Data Science & ML
        'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras',
        'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'tableau', 'power bi',
        'opencv', 'nltk', 'spacy', 'statsmodels', 'scipy',
        
        # Mobile Development
        'android', 'ios', 'flutter', 'react native', 'xamarin', 'cordova', 'phonegap',
        'ionic', 'unity', 'kotlin', 'objective-c',
        
        # Other Tools & Technologies
        'linux', 'ubuntu', 'windows', 'macos', 'jira', 'confluence', 'slack', 'postman',
        'vs code', 'visual studio', 'intellij', 'eclipse', 'vim', 'emacs', 'sublime',
        'figma', 'sketch', 'photoshop', 'illustrator', 'blender', 'unity3d',
        'rabbitmq', 'kafka', 'redis', 'memcached', 'graphql', 'rest', 'soap',
        'json', 'xml', 'yaml', 'csv', 'excel', 'word', 'powerpoint'
    }
    
    # Convert resume to lowercase for matching
    resume_lower = resume_text.lower()
    
    # Find skills mentioned in the resume
    found_skills = []
    for skill in technical_skills:
        # Special handling for single-letter skills like "C"
        if skill.lower() == 'c':
            # Look for "C" as standalone or followed by non-alphanumeric (but not "+")
            pattern = r'\bC\b(?!\+)'
            if re.search(pattern, resume_text):  # Case sensitive for "C"
                found_skills.append('C')
        else:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, resume_lower):
                found_skills.append(skill.title())  # Capitalize for display
    
    # Look for skills in common sections
    skills_sections = re.findall(r'(?:skills?|technologies?|tools?)[:\-\s]*([^\n\r]*(?:\n(?!\s*\n)[^\n\r]*)*)', 
                                resume_lower, re.IGNORECASE | re.MULTILINE)
    
    for section in skills_sections:
        # Extract individual skills from comma-separated or bullet-pointed lists
        items = re.findall(r'[•\-\*]?\s*([a-zA-Z0-9\+\#\.]+(?:\s+[a-zA-Z0-9\+\#\.]+)*)', section)
        for item in items:
            item = item.strip()
            if len(item) > 1 and item.lower() in [s.lower() for s in technical_skills]:
                if item.title() not in found_skills:
                    found_skills.append(item.title())
    
    return list(set(found_skills))  # Remove duplicates


def extract_projects(resume_text: str) -> List[Dict[str, str]]:
    """
    Extract project information from resume text.
    Returns a list of dictionaries with project details.
    """
    projects = []
    
    # Common project section headers
    project_patterns = [
        r'(?:projects?|personal projects?|academic projects?|work projects?)[:\-\s]*\n',
        r'(?:portfolio|github projects?)[:\-\s]*\n'
    ]
    
    # Find project sections
    for pattern in project_patterns:
        matches = re.finditer(pattern, resume_text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            # Get text after the project header
            start_pos = match.end()
            
            # Find the next major section or end of text
            next_section = re.search(r'\n(?:[A-Z][A-Z\s]{3,}|EXPERIENCE|EDUCATION|SKILLS|CERTIFICATIONS)', 
                                   resume_text[start_pos:], re.IGNORECASE)
            
            if next_section:
                project_text = resume_text[start_pos:start_pos + next_section.start()]
            else:
                project_text = resume_text[start_pos:]
            
            # Extract individual projects
            project_entries = extract_individual_projects(project_text)
            projects.extend(project_entries)
    
    # Also look for project-like entries throughout the resume
    additional_projects = find_standalone_projects(resume_text)
    projects.extend(additional_projects)
    
    # Remove duplicates based on project title
    unique_projects = []
    seen_titles = set()
    
    for project in projects:
        title_lower = project['title'].lower().strip()
        if title_lower not in seen_titles and len(title_lower) > 2:
            seen_titles.add(title_lower)
            unique_projects.append(project)
    
    return unique_projects[:10]  # Limit to top 10 projects


def extract_individual_projects(project_text: str) -> List[Dict[str, str]]:
    """Helper function to extract individual projects from a project section."""
    projects = []
    
    # Split by common project separators
    project_blocks = re.split(r'\n(?=\s*[•\-\*]|\s*\d+\.|\s*[A-Z][a-zA-Z\s]+:)', project_text)
    
    for block in project_blocks:
        if len(block.strip()) < 10:  # Skip very short blocks
            continue
            
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
        
        # First line is likely the project title
        title_line = lines[0]
        
        # Clean up title (remove bullets, numbers, etc.)
        title = re.sub(r'^[•\-\*\d\.\s]*', '', title_line).strip()
        title = re.sub(r'[:\-\|].*$', '', title).strip()  # Remove description after colon/dash
        
        if len(title) < 3:
            continue
        
        # Get description from remaining lines
        description_lines = lines[1:] if len(lines) > 1 else []
        description = ' '.join(description_lines)[:200]  # Limit description length
        
        # Extract technologies mentioned in this project
        technologies = extract_project_technologies(block)
        
        projects.append({
            'title': title,
            'description': description,
            'technologies': ', '.join(technologies) if technologies else ''
        })
    
    return projects


def find_standalone_projects(resume_text: str) -> List[Dict[str, str]]:
    """Find project-like entries that might not be in a dedicated project section."""
    projects = []
    
    # Look for lines that might be project titles
    lines = resume_text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip very short or very long lines
        if len(line) < 5 or len(line) > 100:
            continue
        
        # Look for patterns that suggest a project title
        project_indicators = [
            r'\b(?:developed|built|created|implemented|designed)\b.*(?:application|app|website|system|tool)',
            r'\b(?:e-commerce|portfolio|dashboard|chatbot|game)\b',
            r'\b(?:web|mobile|desktop)\s+(?:app|application)\b'
        ]
        
        for pattern in project_indicators:
            if re.search(pattern, line, re.IGNORECASE):
                # Extract a clean title
                title = re.sub(r'^[•\-\*\d\.\s]*', '', line).strip()
                
                # Get some context from surrounding lines
                context_lines = []
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    if j != i and lines[j].strip():
                        context_lines.append(lines[j].strip())
                
                description = ' '.join(context_lines)[:150]
                
                projects.append({
                    'title': title,
                    'description': description,
                    'technologies': ''
                })
                break
    
    return projects


def extract_project_technologies(project_text: str) -> List[str]:
    """Extract technologies mentioned in a specific project description."""
    # Common technical terms to look for in project descriptions
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'html', 'css',
        'mysql', 'mongodb', 'postgresql', 'django', 'flask', 'spring',
        'aws', 'azure', 'docker', 'git', 'github', 'api', 'rest'
    ]
    
    found_tech = []
    project_lower = project_text.lower()
    
    for tech in tech_keywords:
        if re.search(r'\b' + re.escape(tech) + r'\b', project_lower):
            found_tech.append(tech.title())
    
    return found_tech



# ✅ Final Evaluation Prompt Generator
def generate_final_evaluation_prompt(full_text):
    return  f"""
You are SmartestStep AI — a highly experienced communication and knowledge assessment mentor.

You've reviewed thousands of interview transcripts. Your job is to evaluate how well the candidate performed in terms of communication and subject knowledge. Do not penalize based on accent — focus on clarity, structure, tone, and actual content accuracy.

---

### 🧠 Communication Evaluation Criteria:

1. **Tone**
   - Did the candidate sound calm, natural, and professionally engaged?

2. **Clarity & Structure**
   - Were their ideas well-organized and easy to understand?

3. **Confidence**
   - Did they speak with self-assurance and minimal hesitation?

For each, score from 0–10 and label:
- Low: 0–3
- Medium: 4–7
- High: 8–10

---

### 🧪 Knowledge Evaluation:

- Review the transcript and identify the number of questions attempted by the candidate.
- For each response, classify it as:
  - ✅ **Correct**
  - ⚠️ **Partially Correct**
  - ❌ **Incorrect or Unclear**

- Use the following formula:
  - **Answer Accuracy %** = (Correct + 0.5 × Partially Correct) / Total Questions × 100
  - **Answer Accuracy Score (0–10)** = Answer Accuracy % / 10

---

### 🧮 Final Interview Score:

- **Communication Score (0–10)** = Average of Tone, Clarity, Confidence
- **Final Interview Score (0–10)** = Average of Communication Score and Answer Accuracy Score

---

### ✅ Pass/Fail Logic:

- If **Final Interview Score ≥ 9.0 (90%)**:
  ✅ “You have passed this round and may proceed to the next stage.”
- Else:
  ❌ “You did not pass. Please review the feedback and try again.”

---

### 🗣 Input:
Full Interview Transcript (Only Candidate's Spoken Responses):
\"\"\"{full_text}\"\"\"

---

### 📤 Output Format:

**Tone Score**: [0–10], Level: [Low / Medium / High]  
**Clarity & Structure Score**: [0–10], Level: [Low / Medium / High]  
**Confidence Score**: [0–10], Level: [Low / Medium / High]  
**Overall Communication Score**: [0–10]

📚 **Knowledge Evaluation**:  
- Total Questions: [X]  
- Correct Answers: [Y]  
- Partially Correct Answers: [Z]  
- Incorrect Answers: [W]  
- Answer Accuracy Score: [0–10]

🎯 **Final Interview Score**: [0–10]  
✅/❌ **Interview Status**: [Pass or Fail Message]

---

📝 **Feedback Summary**:  
- [Brief summary on strengths and weaknesses]

🔧 **Top 3 Suggestions for Improvement**:  
1. [Tip 1]  
2. [Tip 2]  
3. [Tip 3]
"""




# ✅ LLM processor
# 💡 CHANGED: LLM processor now supports get/set of excluded questions
class LanguageModelProcessor:
    def __init__(
        self,
        resume_text,
        stage="Basic",
        candidate_type="unknown",
        skills_list=None,
        projects_list=None,
        excluded_questions=None  # 💡 NEW
    ):
        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Store excluded questions (from transcript history)
        self.excluded_questions = excluded_questions or set()  # 💡 NEW

        system_prompt = generate_stage_prompt(
            stage,
            resume_text,
            candidate_type,
            skills_list,
            projects_list,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}"),
        ])

        self.conversation = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        start = time.time()
        response = self.conversation.invoke({"text": text})
        self.memory.chat_memory.add_ai_message(response["text"])
        print(f"LLM ({int((time.time() - start) * 1000)}ms): {response['text']}")
        return response["text"]

    def evaluate_final_report(self, full_text):
        prompt = generate_final_evaluation_prompt(full_text)
        start = time.time()
        response = self.conversation.invoke({"text": prompt})
        print(f"Final Evaluation LLM ({int((time.time() - start) * 1000)}ms):\n{response['text']}")
        return response["text"]

    def evaluate_answer(self, question, answer):
        # [UNCHANGED...]
        evaluation_prompt = f"""
                            You are an interview evaluator.
                            Question: {question}
                            Candidate Answer: {answer}

                            1. ✅ Give a correct answer.
                            2. 🧠 Then, evaluate the candidate's answer and classify it with **exactly one** of these three categories (use these exact phrases):
                                - Correct
                                - Partially correct
                                - Incorrect
                            Do NOT use any other words, combinations, or synonyms.
                            3. Provide a short reason for your evaluation.
                            Format:
                            Ideal Answer: ...
                            Answer Check: ...
        """
        response = self.llm.invoke(evaluation_prompt).content.strip()

        # Split response cleanly
        lines = response.split("\n")
        ideal_answer = ""
        correctness = ""

        for line in lines:
            if line.lower().startswith("ideal answer"):
                ideal_answer = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer check"):
                correctness = line.split(":", 1)[1].strip()

        return ideal_answer, correctness
# Updated generate_stage_prompt function with skill and project extraction
def generate_stage_prompt(stage, resume_text, candidate_type="unknown", skills_list=None, projects_list=None):
    # [UNCHANGED: ...as in your code, except ADD instructions to not repeat from previous sessions]
    if skills_list is None:
        skills_list = extract_skills(resume_text)
    if projects_list is None:
        projects_list = extract_projects(resume_text)

    skills_text = ", ".join(skills_list[:8]) if skills_list else "general technical concepts"
    projects_text = ""
    if projects_list:
        projects_text = "\n   - Projects to reference:\n"
        for project in projects_list[:5]:  # Limit to top 5 projects
            projects_text += f"     • {project['title']}"
            if project.get('technologies'):
                projects_text += f" (using {project['technologies']})"
            projects_text += "\n"
    
    return f"""
You are a friendly, professional virtual interviewer conducting the **{stage} Technical Stage** of a technical interview.

🎯 OBJECTIVE:
- Evaluate the candidate's **advanced technical knowledge and practical application**.
- Detect whether the candidate is a **student** or a **working professional**, and adjust tone/wording slightly — but do NOT change question complexity.
- Ask only **advanced-level questions**, deeply tied to the candidate’s listed skills and projects. Avoid theoretical trivia.

📋 SESSION INFO:
- Candidate Type: {candidate_type}
- Key Skills to Focus On: {skills_text}
- Resume Context: {resume_text[:500]}...{projects_text}

---

🧠 QUESTIONING RULES:
1. Ask **exactly one question per turn**.
2. Ask a total of **12–15 advanced-level technical questions**, then conclude the interview.

3. Only ask **advanced questions**, such as:
   - Scenario-based questions (e.g., "How would you handle X if Y occurred?")
   - Architectural or optimization tradeoffs
   - Advanced debugging, design, or deployment issues
   - Questions that explore reasoning, critical thinking, and experience

4. Rotate through **2–3 key technical skills** from these extracted skills: {skills_text}
   - Ask **2–4 advanced questions per skill**.
   - Do NOT ask about any topic, library, or tool not listed in the skills section.

5. Deeply leverage listed projects:
   - Ask **real-world**, **experience-based** questions like:
     - "In project X, how did you address [challenge]?"
     - "How would you scale/improve project Y today?"
     - "What trade-offs did you make when designing feature Z?"
   - Ask 2–3 project-specific questions spaced out across the session.

6. Customize wording based on candidate type:
   👨‍🎓 *Student* → focus on deep understanding and strong reasoning
   👩‍💼 *Professional* → ask workplace-scenario questions or cross-functional impact
   *(Both should face advanced technical difficulty — not behavioral questions)*

---

⚠️ ADVANCED QUESTION TONE REQUIREMENTS:
- Do NOT ask definitions or trivia.
- Do NOT ask basic-level comparisons or conceptual theory.
- Do NOT explain questions — challenge the candidate to reason.
- Do NOT repeat topics/questions from Basic or Intermediate rounds.

---

💬 RESPONSE RULES:

1. For each turn, your reply MUST ONLY consist of the *next technical question* to ask.
2. Do NOT output "Noted.", "Thank you.", "Let's move on", "Let's skip", "That's okay", or any similar transition or acknowledgment phrases.
3. Do NOT recap the previous question or answer.
4. Do NOT provide feedback, encouragement, summaries, or any polite filler.
5. Your output should always be a single, **direct**, **technical**, **advanced-level** question ending with a "?" — nothing else.

---

🚫 ABSOLUTELY DO NOT:
- ❌ Explain or define correct answers
- ❌ Give feedback like "That's right" or "Actually…"
- ❌ Teach or correct any concepts
- ❌ Repeat questions (even with slightly different wording) — across this session or previous ones
- ❌ Ask compound/multi-part questions
- ❌ Use basic or intermediate concepts (e.g., "What is a variable?")
- ❌ Ask HR, behavioral, or soft-skill questions
- ❌ Include transition phrases (e.g., "Let's move on", "That's okay")

---

🛑 RESPONSE HANDLING RULES:

- For a candidate answer:  
  - ✅ On most turns, respond ONLY with the next technical question.
- If the candidate says "I don't know", "I'm not sure", or "Can we skip?":
  - ✅ Respond ONLY with the next technical question (no filler).
- If silent for 10 seconds:
  - ✅ Respond: "Are you still there? Please take your time and answer when you're ready."
  - If still silent after another 10 seconds, directly ask the next technical question (no filler).
- If the candidate asks a question (e.g., "What do you mean?" or "Can you explain?"):
  - ✅ Respond: "Let's focus on the current question. Please answer first — we'll clarify afterward."

---

📌 FEEDBACK POLICY:
- Do NOT explain, correct, or guide the candidate.
- Feedback will be provided only after the full session ends.
- If you begin explaining, immediately stop and say: "Noted." Move to the next question.

---

🏁 ENDING INSTRUCTIONS:
After 12–15 questions, end with:
**"We've completed the {stage} Technical Stage. Thank you for your time. Your responses will now be reviewed."**
"""



# ✅ Load latest resume text file
def get_latest_resume_text():
    resume_folder = "resumes"
    if not os.path.exists(resume_folder):
        raise Exception("❌ 'resumes' folder does not exist.")

    txt_files = [f for f in os.listdir(resume_folder) if f.endswith(".txt")]
    if not txt_files:
        raise Exception("❌ No .txt resume found in the 'resumes' folder.")

    txt_files.sort(key=lambda f: os.path.getmtime(os.path.join(resume_folder, f)), reverse=True)
    latest_file = os.path.join(resume_folder, txt_files[0])

    print(f"✅ Latest resume detected: {latest_file}")
    with open(latest_file, "r", encoding="utf-8") as f:
        return f.read().strip()


# ✅ Detect candidate type
def detect_candidate_type(resume_text):
    student_keywords = ["student", "undergraduate", "pursuing", "bachelor", "b.tech", "bsc", "bca", "college", "university", "currently studying"]
    professional_keywords = ["experience", "worked", "developer", "engineer", "software", "company", "organization", "intern", "employment"]

    student_score = sum(kw in resume_text.lower() for kw in student_keywords)
    professional_score = sum(kw in resume_text.lower() for kw in professional_keywords)

    if student_score > professional_score:
        return "student"
    elif professional_score > student_score:
        return "professional"
    elif student_score == professional_score and student_score > 0:
        return "both"
    else:
        return "unknown"



# ✅ Transcript helper
class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return " ".join(self.transcript_parts)

transcript_collector = TranscriptCollector()

# ✅ Corrected get_transcript function
async def get_transcript(callback):
    try:
        transcription_complete = asyncio.Event()

        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        loop = asyncio.get_running_loop()

        async def on_message(connection, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript

            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript().strip()
                if full_sentence:
                    print(f"User: {full_sentence}")
                    await callback(full_sentence)
                    transcript_collector.reset()
                    transcription_complete.set()

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=500,
            smart_format=True,
        )

        await dg_connection.start(options)

        microphone = Microphone(
            lambda data: loop.call_soon_threadsafe(
                lambda: asyncio.create_task(dg_connection.send(data))
            )
        )
        microphone.start()

        await transcription_complete.wait()

        microphone.finish()
        await dg_connection.finish()

    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")

# Set your user_id here for now. Change this when integrating a frontend.
USER_ID = "test_candidate"  # <---- substitute with "candidate_email@example.com" later

class ConversationManager:
    def __init__(self, resume_text, stage="Basic", candidate_type="unknown", user_id="test_candidate"):
        self.full_transcript_list = []
        self.human_responses = []
        self.resume_text = resume_text
        self.user_id = user_id
        print(f"🔑 User ID for this session: {self.user_id}")

        self.transcript_filename = f"transcripts/interview_{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("transcripts", exist_ok=True)

        with open(self.transcript_filename, "w", encoding="utf-8") as f:
            f.write(f"=== INTERVIEW SESSION STARTED ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stage: {stage}\n")
            f.write(f"Candidate Type: {candidate_type}\n")
            f.write("="*50 + "\n\n")

        self.previous_questions = get_all_previous_questions(self.user_id)
        self.llm = LanguageModelProcessor(
            resume_text=resume_text,
            stage=stage,
            candidate_type=candidate_type,
            excluded_questions=self.previous_questions
        )

        self.transcript_queue = asyncio.Queue()
        print(f"🧠 Resume Loaded. Beginning {stage} Stage Interview...")
        print(f"📝 Transcript will be saved to: {self.transcript_filename}\n")
        # The first welcome/question is generated here
        self.llm.process("Hello, I am ready to begin the interview.")

    async def listen_and_respond(self):
        async def handle_full_sentence(full_sentence):
            await self.transcript_queue.put(full_sentence)

        while True:
            # Save current AI question BEFORE listening, so it's always the exact prompt given to the user.
            ai_question_for_transcript = ""
            if self.llm.memory.chat_memory.messages:
                ai_question_for_transcript = self.llm.memory.chat_memory.messages[-1].content.strip()

            print("🎙️ Listening for speech...")
            await get_transcript(handle_full_sentence)

            while not self.transcript_queue.empty():
                transcript = await self.transcript_queue.get()
                if not transcript:
                    print("⚠️ No transcription received. Retrying...")
                    continue

                if "goodbye" in transcript.lower():
                    print("👋 Goodbye detected. Generating final report and ending session.")
                    await self.generate_report_and_exit()
                    return

                self.human_responses.append(transcript)
                
                ai_response = self.llm.process(transcript)
                # We add the question just asked to excluded set for this session
                self.llm.excluded_questions.add(ai_question_for_transcript)
                ideal_answer, correctness = self.llm.evaluate_answer(ai_question_for_transcript, transcript)

                entry = f"""🔹AI Question: {ai_question_for_transcript}
👤 User Answer: {transcript}
✅ Ideal Answer: {ideal_answer}
🧠 Answer Check: {correctness}
{"-"*50}
"""
                if entry not in self.full_transcript_list:
                    self.full_transcript_list.append(entry)
                    try:
                        with open(self.transcript_filename, "a", encoding="utf-8") as f:
                            f.write(entry + "\n")
                    except Exception as e:
                        print(f"❌ Failed to save Q&A: {e}")

    async def generate_report_and_exit(self):
        unique_responses = list(dict.fromkeys(self.human_responses))
        print("🔄 Generating final evaluation report...")
        final_report = self.llm.evaluate_final_report("\n".join(unique_responses))
        
        try:
            with open(self.transcript_filename, "a", encoding="utf-8") as f:
                f.write("\n" + "="*60 + "\n")
                f.write("📊 FINAL EVALUATION REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(final_report)
                f.write(f"\n\nSession completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"✅ Complete interview transcript saved to: {self.transcript_filename}")
            print("\n" + "="*60)
            print("📊 FINAL EVALUATION REPORT")
            print("="*60)
            print(final_report)
        except Exception as e:
            print(f"❌ Failed to save final report: {e}")

        if os.path.exists("final.txt"):
            try:
                os.remove("final.txt")
                print("🧹 Cleaned up temporary final.txt file")
            except Exception as e:
                print(f"⚠️ Could not remove final.txt: {e}")

    async def main(self):
        await self.listen_and_respond()
        if not self.full_transcript_list:
            print("⚠️ No transcript collected. Skipping report.")
            return

   


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    try:
        resume_text = get_latest_resume_text()
        candidate_type = detect_candidate_type(resume_text)
        print(f"🧠 Detected Candidate Type: {candidate_type}")
        user_id = USER_ID
        manager = ConversationManager(resume_text, stage="Basic", candidate_type=candidate_type,user_id=user_id)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(manager.main())
    except Exception as err:
>>>>>>> a3a8f41ace40ec7c4d8cdbfbd22f04fc9af364be
        print(f"❌ {err}")