from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import os
import asyncio
import threading
import uuid
from datetime import datetime
import json

# Import your existing classes (keep all the existing imports and classes)
from interview_processor import (
    ConversationManager, 
    get_latest_resume_text, 
    detect_candidate_type,
    RAGResumeProcessor,
    extract_skills,
    extract_projects
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for active sessions
active_sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-resume', methods=['POST'])
def upload_resume():
    """Upload and process resume"""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save resume
        os.makedirs('resumes', exist_ok=True)
        filename = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join('resumes', filename)
        
        # Convert to text and save
        resume_text = file.read().decode('utf-8')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(resume_text)
        
        # Process resume
        candidate_type = detect_candidate_type(resume_text)
        skills = extract_skills(resume_text)
        projects = extract_projects(resume_text)
        
        return jsonify({
            'success': True,
            'candidate_type': candidate_type,
            'skills_count': len(skills),
            'projects_count': len(projects),
            'resume_preview': resume_text[:500] + '...' if len(resume_text) > 500 else resume_text
        })
        
    except Exception as e:
        return jsonify({'error': f'Resume processing failed: {str(e)}'}), 500

@app.route('/start-interview', methods=['POST'])
def start_interview():
    """Initialize interview session"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', f"user_{uuid.uuid4().hex[:8]}")
        stage = data.get('stage', 'Basic')
        
        # Get resume
        resume_text = get_latest_resume_text()
        candidate_type = detect_candidate_type(resume_text)
        
        # Create session
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session['user_id'] = user_id
        
        # Initialize conversation manager
        manager = ConversationManager(
            resume_text=resume_text,
            stage=stage,
            candidate_type=candidate_type,
            user_id=user_id
        )
        
        active_sessions[session_id] = {
            'manager': manager,
            'user_id': user_id,
            'stage': stage
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Interview session initialized'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start interview: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to interview system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_recording')
def handle_start_recording(data):
    """Start speech recognition"""
    session_id = data.get('session_id')
    if session_id not in active_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    emit('recording_status', {'status': 'listening'})

@socketio.on('audio_data')
def handle_audio_data(data):
    """Process audio data from client"""
    session_id = data.get('session_id')
    audio_data = data.get('audio_data')
    
    if session_id not in active_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    # Process audio with Deepgram (you'll need to adapt this)
    # For now, simulate transcription
    transcript = process_audio_with_deepgram(audio_data)
    
    if transcript:
        emit('transcript', {'text': transcript})
        
        # Process with LLM
        manager = active_sessions[session_id]['manager']
        response = process_interview_response(manager, transcript)
        
        emit('ai_response', {'text': response})

@socketio.on('text_input')
def handle_text_input(data):
    """Handle text input instead of speech"""
    session_id = data.get('session_id')
    text = data.get('text')
    
    if session_id not in active_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    manager = active_sessions[session_id]['manager']
    
    # Check for goodbye
    goodbye_phrases = ["goodbye", "bye", "end this", "stop", "finish"]
    if any(phrase in text.lower() for phrase in goodbye_phrases):
        final_report = generate_final_report(manager)
        emit('interview_complete', {'report': final_report})
        cleanup_session(session_id)
        return
    
    # Process response
    ai_response = manager.llm.process(text)
    
    # Save to transcript
    save_to_transcript(manager, text, ai_response)
    
    # Check if interview is complete
    END_MARKERS = [
        "We've completed the Basic Technical Stage",
        "We've completed the Intermediate Technical Stage", 
        "We've completed the Advanced Technical Stage",
        "Your responses will now be reviewed."
    ]
    
    if any(marker in ai_response for marker in END_MARKERS):
        final_report = generate_final_report(manager)
        emit('interview_complete', {'report': final_report})
        cleanup_session(session_id)
        return
    
    emit('ai_response', {'text': ai_response})

def process_audio_with_deepgram(audio_data):
    """Process audio with Deepgram API"""
    # You'll need to implement the actual Deepgram processing here
    # This is a placeholder
    return "Sample transcription"

def process_interview_response(manager, transcript):
    """Process interview response"""
    try:
        return manager.llm.process(transcript)
    except Exception as e:
        print(f"Error processing response: {e}")
        return "I apologize, there was an error processing your response. Please try again."

def save_to_transcript(manager, user_input, ai_response):
    """Save conversation to transcript"""
    try:
        entry = f"""🔹AI Question: {ai_response}
👤 User Answer: {user_input}
{"-"*50}
"""
        with open(manager.transcript_filename, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception as e:
        print(f"Failed to save transcript: {e}")

def generate_final_report(manager):
    """Generate final evaluation report"""
    try:
        with open(manager.transcript_filename, "r", encoding="utf-8") as f:
            full_transcript = f.read()
        
        final_report = manager.llm.evaluate_final_report(full_transcript)
        
        # Save final report
        with open(manager.transcript_filename, "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write("📊 FINAL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(final_report)
        
        return final_report
    except Exception as e:
        return f"Error generating report: {str(e)}"

def cleanup_session(session_id):
    """Clean up session data"""
    if session_id in active_sessions:
        del active_sessions[session_id]

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
