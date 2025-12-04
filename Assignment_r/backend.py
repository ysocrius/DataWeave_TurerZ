"""
FastAPI Backend for AI Document Processor
Provides REST API endpoints for the React frontend
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
import tempfile
import os
import time
import json
import asyncio
import re
from typing import Dict, Any, AsyncGenerator, List, Tuple
from PyPDF2 import PdfReader

# Import our existing pipeline
from pdf_to_excel_pipeline import extract_text_from_pdf, process_with_llm
from utils import clean_and_parse_json
from utils_dedup import full_deduplication_pipeline
from config import Config
import pandas as pd

# Import learning system
from performance_tracker import get_performance_tracker
from learning_scheduler import start_scheduler, stop_scheduler, get_scheduler_status

# Character-based Chunking Functions
def sort_entries_logically(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort entries in logical order for professional profiles/resumes
    """
    def get_sort_priority(entry):
        key = (entry.get('Key') or '').lower()
        
        # Personal info (highest priority)
        if any(word in key for word in ['full name', 'name', 'birthdate', 'birth', 'age', 'blood', 'nationality']):
            return (1, key)
        
        # Current employment
        elif any(word in key for word in ['current company', 'current position', 'current salary', 'current start']):
            return (2, key)
        
        # Previous employment
        elif any(word in key for word in ['previous company', 'previous position', 'previous salary', 'previous start', 'previous end', 'previous joining', 'previous starting']):
            return (3, key)
        
        # First job
        elif any(word in key for word in ['first company', 'first position', 'first salary', 'first job']):
            return (4, key)
        
        # Education
        elif any(word in key for word in ['graduation', 'cgpa', 'degree', 'college', 'university', 'school', 'b-tech', 'hi-tech', 'class rank', 'overall score', 'thesis']):
            return (5, key)
        
        # Certifications
        elif any(word in key for word in ['certification', 'aws', 'azure', 'safe', 'pmp', 'project management']):
            return (6, key)
        
        # Skills
        elif any(word in key for word in ['proficiency', 'skill', 'python', 'sql', 'visualization', 'cloud', 'experience']):
            return (7, key)
        
        # Everything else
        else:
            return (8, key)
    
    return sorted(entries, key=get_sort_priority)

def create_character_chunks(full_text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Create character-based chunks with overlap
    """
    # Use config values if not specified
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if overlap is None:
        overlap = Config.CHUNK_OVERLAP
    
    chunks = []
    text_length = len(full_text)
    step_size = chunk_size - overlap
    
    chunk_id = 1
    start = 0
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Get chunk content
        chunk_content = full_text[start:end].strip()
        
        if len(chunk_content) > 50:  # Only include substantial chunks
            # Try to end at sentence boundary if not at document end
            if end < text_length:
                # Look for sentence endings (.!?) within last 100 characters
                last_100 = chunk_content[-100:] if len(chunk_content) > 100 else chunk_content
                sentence_end = max(
                    last_100.rfind('. '),
                    last_100.rfind('! '),
                    last_100.rfind('? '),
                    last_100.rfind('.\n'),
                    last_100.rfind('!\n'),
                    last_100.rfind('?\n')
                )
                
                if sentence_end > len(last_100) * 0.5:  # Only if in last 50%
                    offset = len(chunk_content) - len(last_100)
                    cutoff = offset + sentence_end + 1
                    chunk_content = chunk_content[:cutoff].strip()
                    end = start + cutoff
                elif not chunk_content.endswith(' '):
                    # Fallback: try to end at word boundary
                    last_space = chunk_content.rfind(' ', max(0, len(chunk_content) - 50))
                    if last_space > len(chunk_content) * 0.8:
                        chunk_content = chunk_content[:last_space].strip()
                        end = start + last_space
            
            chunks.append({
                'chunk_id': chunk_id,
                'start_pos': start,
                'end_pos': end,
                'content': chunk_content,
                'length': len(chunk_content),
                'char_range': f"{start+1}-{end}"
            })
            
            chunk_id += 1
        
        start += step_size
        
        # Prevent infinite loop
        if start >= text_length:
            break
    
    return chunks

# Intelligent Semantic Chunking Functions
def create_intelligent_chunks(full_text: str, total_pages: int) -> List[Dict[str, Any]]:
    """
    Create intelligent semantic chunks based on document structure
    """
    chunks = []
    
    # Split text into lines for analysis
    lines = full_text.split('\n')
    
    # Define section patterns (more comprehensive)
    section_patterns = [
        r'^[A-Z][A-Z\s&]{10,}$',  # ALL CAPS headers
        r'^\d+\.\s+[A-Z]',        # Numbered sections
        r'^[A-Z][a-z]+:',         # Title: format
        r'^Event\s+Details?:?',   # Event details
        r'^Schedule:?',           # Schedule sections
        r'^Registration:?',       # Registration info
        r'^Contact:?',            # Contact info
        r'^Hotel:?',              # Hotel info
        r'^Pricing:?',            # Pricing info
        r'^Day\s+\d+',           # Day sections
        r'^Session\s+\d+',       # Session sections
    ]
    
    # Find section boundaries
    section_starts = []
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if len(line_clean) > 3:
            for pattern in section_patterns:
                if re.match(pattern, line_clean, re.IGNORECASE):
                    section_starts.append((i, line_clean))
                    break
    
    # If no clear sections found, create fallback sections
    if len(section_starts) < 3:
        # Create 5-8 sections based on content length
        total_lines = len(lines)
        target_sections = min(8, max(5, total_lines // 50))  # 5-8 sections
        section_size = total_lines // target_sections
        
        section_starts = []
        for i in range(0, total_lines, section_size):
            if i < total_lines:
                # Find a good header line near this position
                header_line = "Content Section"
                for j in range(max(0, i-5), min(total_lines, i+10)):
                    if j < len(lines) and len(lines[j].strip()) > 5:
                        header_line = lines[j].strip()[:50]
                        break
                section_starts.append((i, header_line))
    
    # Create chunks from sections
    for i, (start_line, header) in enumerate(section_starts):
        # Determine end line
        if i + 1 < len(section_starts):
            end_line = section_starts[i + 1][0]
        else:
            end_line = len(lines)
        
        # Extract section content
        section_lines = lines[start_line:end_line]
        section_content = '\n'.join(section_lines).strip()
        
        if len(section_content) > 100:  # Only include substantial sections
            # Add 15% overlap with previous section
            if i > 0 and len(section_lines) > 10:
                overlap_lines = max(2, len(section_lines) // 7)  # ~15% overlap
                prev_end = section_starts[i][0]
                overlap_start = max(0, prev_end - overlap_lines)
                overlap_content = '\n'.join(lines[overlap_start:prev_end])
                if overlap_content.strip():
                    section_content = overlap_content + '\n' + section_content
            
            # Estimate page range
            chars_per_page = len(full_text) / max(1, total_pages)
            section_start_pos = sum(len(line) + 1 for line in lines[:start_line])
            section_end_pos = section_start_pos + len(section_content)
            
            start_page = max(1, int(section_start_pos / chars_per_page) + 1)
            end_page = max(start_page, int(section_end_pos / chars_per_page) + 1)
            start_page = min(start_page, total_pages)
            end_page = min(end_page, total_pages)
            
            chunks.append({
                'chunk_id': i + 1,
                'content': section_content,
                'header': header,
                'section_type': 'semantic_section',
                'start_line': start_line,
                'end_line': end_line,
                'length': len(section_content),
                'start_page': start_page,
                'end_page': end_page,
                'page_range': f"{start_page}-{end_page}" if start_page != end_page else str(start_page)
            })
    
    return chunks



app = FastAPI(title="AI Document Processor API")

@app.get("/")
async def root():
    """Root endpoint for basic connectivity test"""
    return {"message": "AI Document Processor API", "status": "running"}

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event: Start learning scheduler
@app.on_event("startup")
async def startup_event():
    """Start learning scheduler on app startup"""
    if Config.LEARNING_ENABLED:
        try:
            start_scheduler()
            print("✅ Learning scheduler started")
        except Exception as e:
            print(f"⚠️  Failed to start learning scheduler: {e}")

# Shutdown event: Stop learning scheduler
@app.on_event("shutdown")
async def shutdown_event():
    """Stop learning scheduler on app shutdown"""
    try:
        stop_scheduler()
        print("✅ Learning scheduler stopped")
    except Exception as e:
        print(f"⚠️  Failed to stop learning scheduler: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Document Processor API is running"}







@app.post("/api/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Process a PDF file using character-based chunking with real-time SSE progress
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    content = await file.read()
    
    # Initialize performance tracker
    tracker = get_performance_tracker()
    session_id = tracker.create_session_id()
    
    async def generate_events() -> AsyncGenerator[str, None]:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_pdf_path = temp_file.name
        
        try:
            start_time = time.time()
            
            # Phase 1: Text Extraction & Character Analysis
            yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'extracting_text'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Extract full text and get page count
            full_text = extract_text_from_pdf(temp_pdf_path)
            reader = PdfReader(temp_pdf_path)
            total_pages = len(reader.pages)
            
            # Create character-based chunks
            char_chunks = create_character_chunks(full_text)
            total_chunks = len(char_chunks)
            
            yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'character_chunking', 'chunk_size': Config.CHUNK_SIZE, 'overlap': Config.CHUNK_OVERLAP})}\n\n"
            await asyncio.sleep(0.1)
            
            # Phase 2: Character Chunk Processing
            chunk_results = []
            processed_chunks = []
            
            for i, chunk in enumerate(char_chunks):
                chunk_num = i + 1
                
                # Calculate approximate page range
                chars_per_page = len(full_text) / max(1, total_pages)
                start_page = max(1, int(chunk['start_pos'] / chars_per_page) + 1)
                end_page = max(start_page, int(chunk['end_pos'] / chars_per_page) + 1)
                start_page = min(start_page, total_pages)
                end_page = min(end_page, total_pages)
                page_range = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)
                
                # Send chunk started event
                yield f"data: {json.dumps({'event': 'chunk_start', 'chunk': chunk_num, 'total_chunks': total_chunks, 'char_range': chunk['char_range'], 'page_range': page_range, 'status': 'started'})}\n\n"
                await asyncio.sleep(0.1)
                
                chunk_start_time = time.time()
                
                # Send processing events
                yield f"data: {json.dumps({'event': 'chunk_progress', 'chunk': chunk_num, 'total_chunks': total_chunks, 'status': 'processing'})}\n\n"
                await asyncio.sleep(0.1)
                
                try:
                    # Process chunk with LLM using expected format
                    from prompt_templates import get_expected_format_prompt
                    from openai import OpenAI
                    import os
                    
                    # Get API credentials
                    api_key = os.getenv("OPENAI_API_KEY")
                    model_name = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
                    client = OpenAI(api_key=api_key)
                    
                    # Use expected format prompt
                    prompt = get_expected_format_prompt(chunk['content'])
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    raw_llm_response = response.choices[0].message.content
                    parsed_data = clean_and_parse_json(raw_llm_response)
                    chunk_entries = parsed_data.get('entries', [])
                    
                    chunk_end_time = time.time()
                    chunk_duration = round(chunk_end_time - chunk_start_time, 2)
                    
                    chunk_info = {
                        "chunk_id": chunk_num,
                        "char_range": chunk['char_range'],
                        "page_range": page_range,
                        "entries_count": len(chunk_entries),
                        "processing_time": chunk_duration,
                        "content_length": chunk['length'],
                        "status": "success"
                    }
                    
                    processed_chunks.append(chunk_info)
                    chunk_results.append({
                        "chunk_id": chunk_num,
                        "entries": chunk_entries,
                        "global_notes": parsed_data.get('global_notes')
                    })
                    
                    # Send chunk complete event
                    yield f"data: {json.dumps({'event': 'chunk_complete', 'chunk': chunk_num, 'total_chunks': total_chunks, 'char_range': chunk['char_range'], 'entries_count': len(chunk_entries), 'processing_time': chunk_duration, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    chunk_end_time = time.time()
                    chunk_duration = round(chunk_end_time - chunk_start_time, 2)
                    
                    processed_chunks.append({
                        "chunk_id": chunk_num,
                        "char_range": chunk['char_range'],
                        "page_range": page_range,
                        "entries_count": 0,
                        "processing_time": chunk_duration,
                        "status": "error",
                        "error": str(e)
                    })
                    
                    # Send chunk error event
                    yield f"data: {json.dumps({'event': 'chunk_complete', 'chunk': chunk_num, 'total_chunks': total_chunks, 'char_range': chunk['char_range'], 'entries_count': 0, 'processing_time': chunk_duration, 'status': 'error', 'error': str(e)})}\n\n"
                    await asyncio.sleep(0.1)
            
            # Phase 3: Enhanced Deduplication with Fuzzy Matching
            yield f"data: {json.dumps({'event': 'merging_start', 'status': 'deduplicating'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Collect all entries from chunks
            all_entries = []
            global_notes = None
            
            for chunk_result in chunk_results:
                entries = chunk_result.get('entries', [])
                all_entries.extend(entries)
                if not global_notes and chunk_result.get('global_notes'):
                    global_notes = chunk_result.get('global_notes')
            
            initial_count = len(all_entries)
            
            # Run full deduplication pipeline
            yield f"data: {json.dumps({'event': 'dedup_progress', 'status': 'running_deduplication', 'initial_count': initial_count})}\n\n"
            await asyncio.sleep(0.1)
            
            deduplicated_entries, dedup_stats = full_deduplication_pipeline(
                all_entries,
                fuzzy_threshold=Config.FUZZY_MATCH_THRESHOLD,
                enable_fuzzy=Config.ENABLE_FUZZY_DEDUP,
                enable_repair=Config.ENABLE_TRUNCATION_REPAIR
            )
            
            # Send deduplication stats
            yield f"data: {json.dumps({'event': 'dedup_complete', 'stats': dedup_stats})}\n\n"
            await asyncio.sleep(0.1)
            
            # Sort and add row numbers
            sorted_entries = sort_entries_logically(deduplicated_entries)
            for i, entry in enumerate(sorted_entries, 1):
                entry['#'] = i
            
            # Add global notes to first entry
            if global_notes and len(sorted_entries) > 0:
                first_entry = sorted_entries[0]
                existing_comment = first_entry.get('Comments') or ''
                if existing_comment:
                    first_entry['Comments'] = f"{global_notes} | {existing_comment}"
                else:
                    first_entry['Comments'] = global_notes
            
            end_time = time.time()
            total_duration = round(end_time - start_time, 2)
            
            # Track performance in learning system
            if Config.LEARNING_ENABLED:
                try:
                    tracker.track_processing_session(
                        session_id=session_id,
                        filename=file.filename,
                        file_size=len(content),
                        total_pages=total_pages,
                        chunk_size=Config.CHUNK_SIZE,
                        chunk_overlap=Config.CHUNK_OVERLAP,
                        llm_model=Config.LLM_MODEL,
                        temperature=Config.LLM_TEMPERATURE,
                        fuzzy_threshold=Config.FUZZY_MATCH_THRESHOLD,
                        chunk_results=processed_chunks,
                        dedup_stats=dedup_stats,
                        final_entries_count=len(sorted_entries),
                        total_processing_time=total_duration,
                        status="success"
                    )
                except Exception as e:
                    print(f"⚠️  Learning system tracking failed: {e}")
            
            # Send complete event
            final_result = {
                "event": "character_complete",
                "session_id": session_id,
                "entries": sorted_entries,
                "global_notes": global_notes,
                "chunks": processed_chunks,
                "chunk_results": chunk_results,
                "dedup_stats": dedup_stats,
                "total_rows": len(sorted_entries),
                "total_pages": total_pages,
                "total_chunks": len(processed_chunks),
                "processing_time": total_duration,
                "processing_strategy": "character_based_chunking",
                "chunk_size": Config.CHUNK_SIZE,
                "overlap": Config.CHUNK_OVERLAP
            }
            yield f"data: {json.dumps(final_result)}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        
        finally:
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



@app.post("/api/download-excel")
async def download_excel(data: Dict[str, Any]):
    """
    Convert processed data to Excel and return file
    
    Args:
        data: Processed data with entries and global_notes
        
    Returns:
        Excel file download
    """
    try:
        entries = data.get('entries', [])
        global_notes = data.get('global_notes', None)
        
        # Create DataFrame
        df = pd.DataFrame(entries)
        
        # Handle global notes
        if global_notes and global_notes.strip():
            if 'Comments' not in df.columns:
                df['Comments'] = None
            first_comment = df.loc[0, 'Comments']
            if pd.isna(first_comment) or not first_comment:
                df.loc[0, 'Comments'] = f"[GLOBAL NOTE: {global_notes}]"
            else:
                df.loc[0, 'Comments'] = f"[GLOBAL NOTE: {global_notes}] | {first_comment}"
        
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            excel_path = temp_file.name
            df.to_excel(excel_path, index=False, engine='openpyxl')
        
        # Return file
        return FileResponse(
            excel_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename='AI_Extracted_Data.xlsx',
            background=lambda: os.unlink(excel_path) if os.path.exists(excel_path) else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel generation failed: {str(e)}")

@app.get("/api/feedback/{session_id}")
async def get_feedback(session_id: str):
    """
    Get feedback for a specific session (including auto-generated feedback)
    
    Args:
        session_id: Session ID to fetch feedback for
    """
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        learning_sys = get_performance_tracker().learning_system
        
        if not learning_sys.is_connected():
            return {"success": False, "message": "Learning system not connected"}
        
        # Query feedback collection for this session
        from config import Config as ConfigModule
        feedback_collection = learning_sys.db[f"{ConfigModule.MONGODB_COLLECTION_PREFIX}user_feedback"]
        
        # Find the most recent feedback for this session
        feedback_doc = feedback_collection.find_one(
            {"session_id": session_id},
            sort=[("timestamp", -1)]  # Get most recent
        )
        
        if feedback_doc:
            # Remove MongoDB _id field
            feedback_doc.pop('_id', None)
            return {"success": True, "feedback": feedback_doc}
        else:
            return {"success": False, "message": "No feedback found for this session"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feedback: {str(e)}")


@app.post("/api/feedback")
async def submit_feedback(feedback_data: Dict[str, Any]):
    """
    Submit user feedback for a processing session
    
    Args:
        feedback_data: {
            "session_id": str,
            "rating": int (1-5),
            "feedback_text": str (optional),
            "corrections": list (optional)
        }
    """
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from learning_models import UserFeedback
        import uuid
        
        feedback = UserFeedback(
            feedback_id=f"feedback_{uuid.uuid4().hex[:16]}",
            session_id=feedback_data.get('session_id'),
            rating=feedback_data.get('rating', 3),
            feedback_text=feedback_data.get('feedback_text'),
            corrections=feedback_data.get('corrections', []),
            corrections_count=len(feedback_data.get('corrections', []))
        )
        
        learning_sys = get_performance_tracker().learning_system
        success = learning_sys.store_user_feedback(feedback)
        
        if success:
            # Analyze feedback for learning if there's a comment
            feedback_text = feedback_data.get('feedback_text', '').strip()
            if feedback_text:
                try:
                    from feedback_analyzer import feedback_analyzer
                    
                    # Get auto-feedback for comparison
                    feedback_collection = learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}user_feedback"]
                    auto_feedback = feedback_collection.find_one({
                        'session_id': feedback_data.get('session_id'),
                        'feedback_id': {'$regex': '^auto_'}
                    })
                    
                    auto_rating = auto_feedback.get('rating', 0) if auto_feedback else 0
                    
                    # Analyze the comment
                    analysis = feedback_analyzer.analyze_comment(
                        comment=feedback_text,
                        user_rating=feedback_data.get('rating', 3),
                        auto_rating=auto_rating,
                        session_id=feedback_data.get('session_id')
                    )
                    
                    if analysis.get('has_content'):
                        print(f"🧠 Feedback Analysis: {analysis.get('disagreement_type', 'unknown')}")
                        print(f"   Issues: {', '.join(analysis.get('issues', []))}")
                        print(f"   Sentiment: {analysis.get('sentiment', 0):.2f}")
                        
                        # Store analysis for future algorithm improvements
                        results = feedback_analyzer.generate_improvement_suggestions([analysis])
                        if results.get('suggestions'):
                            feedback_analyzer.store_analysis_results(results)
                            print(f"   Suggestions: {len(results['suggestions'])} improvement suggestions generated")
                
                except Exception as e:
                    print(f"⚠️  Feedback analysis failed: {e}")
            
            return {"success": True, "message": "Feedback submitted successfully"}
        else:
            return {"success": False, "message": "Failed to store feedback"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/api/analytics")
async def get_analytics():
    """Get system analytics and performance metrics"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False, "message": "Learning system is disabled"}
    
    try:
        learning_sys = get_performance_tracker().learning_system
        
        if not learning_sys.is_connected():
            return {"enabled": False, "message": "Learning system not connected"}
        
        analytics = learning_sys.get_system_analytics()
        trends = learning_sys.get_performance_trends(days=7)
        
        return {
            "enabled": True,
            "analytics": analytics,
            "trends": trends
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@app.get("/api/session/{session_id}")
async def get_session_details(session_id: str):
    """Get details for a specific processing session"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False, "message": "Learning system is disabled"}
    
    try:
        tracker = get_performance_tracker()
        summary = tracker.get_performance_summary(session_id)
        
        if summary:
            return {"success": True, "session": summary}
        else:
            return {"success": False, "message": "Session not found"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session retrieval failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Simple health check for deployment"""
    try:
        health = {
            "status": "ok",
            "api": "running",
            "timestamp": time.time()
        }
        
        # Only check learning system if enabled and avoid errors
        if Config.LEARNING_ENABLED:
            try:
                learning_sys = get_performance_tracker().learning_system
                health["learning_enabled"] = True
                health["learning_connected"] = learning_sys.is_connected()
            except Exception as e:
                health["learning_enabled"] = True
                health["learning_error"] = str(e)
                health["learning_connected"] = False
        else:
            health["learning_enabled"] = False
        
        return health
    except Exception as e:
        return {
            "status": "error",
            "api": "running",
            "error": str(e),
            "timestamp": time.time()
        }


@app.post("/api/learning/learn-patterns")
async def trigger_pattern_learning():
    """Manually trigger pattern learning"""
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from learning_scheduler import trigger_pattern_learning
        trigger_pattern_learning()
        return {"success": True, "message": "Pattern learning triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern learning failed: {str(e)}")


@app.post("/api/learning/detect-optimizations")
async def trigger_optimization_detection():
    """Manually trigger optimization detection"""
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from learning_scheduler import trigger_optimization_detection
        trigger_optimization_detection()
        return {"success": True, "message": "Optimization detection triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization detection failed: {str(e)}")


@app.get("/api/learning/patterns")
async def get_learned_patterns():
    """Get all learned patterns"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False, "patterns": []}
    
    try:
        learning_sys = get_performance_tracker().learning_system
        
        if not learning_sys.is_connected():
            return {"enabled": False, "message": "Learning system not connected"}
        
        # Get all patterns
        patterns = list(learning_sys.collections['patterns'].find().sort('confidence_score', -1))
        
        # Convert ObjectId to string for JSON serialization
        for pattern in patterns:
            pattern['_id'] = str(pattern['_id'])
        
        return {
            "enabled": True,
            "patterns": patterns,
            "count": len(patterns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patterns: {str(e)}")


@app.get("/api/learning/optimizations")
async def get_active_optimizations():
    """Get active optimizations"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False, "optimizations": []}
    
    try:
        learning_sys = get_performance_tracker().learning_system
        
        if not learning_sys.is_connected():
            return {"enabled": False, "message": "Learning system not connected"}
        
        # Get active optimizations
        optimizations = list(learning_sys.collections['optimizations'].find({
            'is_active': True
        }).sort('timestamp', -1))
        
        # Convert ObjectId to string for JSON serialization
        for opt in optimizations:
            opt['_id'] = str(opt['_id'])
        
        return {
            "enabled": True,
            "optimizations": optimizations,
            "count": len(optimizations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve optimizations: {str(e)}")


@app.get("/api/learning/scheduler-status")
async def get_learning_scheduler_status():
    """Get scheduler status"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False}
    
    try:
        status = get_scheduler_status()
        return {
            "enabled": True,
            "scheduler": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler status: {str(e)}")


@app.post("/api/learning/analyze-feedback")
async def analyze_user_feedback():
    """Manually trigger feedback analysis"""
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from feedback_analyzer import analyze_user_feedback
        results = analyze_user_feedback()
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback analysis failed: {str(e)}")


@app.get("/api/learning/feedback-analysis")
async def get_feedback_analysis_history():
    """Get feedback analysis history"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False, "analyses": []}
    
    try:
        learning_sys = get_performance_tracker().learning_system
        
        if not learning_sys.is_connected():
            return {"enabled": False, "message": "Learning system not connected"}
        
        # Get recent feedback analyses
        analysis_collection = learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}feedback_analysis"]
        analyses = list(analysis_collection.find().sort('timestamp', -1).limit(10))
        
        # Convert ObjectId to string for JSON serialization
        for analysis in analyses:
            analysis['_id'] = str(analysis['_id'])
        
        return {
            "enabled": True,
            "analyses": analyses,
            "count": len(analyses)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback analyses: {str(e)}")


@app.post("/api/learning/run-full-cycle")
async def run_full_learning_cycle():
    """Run complete learning cycle: analyze → recognize patterns → tune algorithm"""
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from learning_orchestrator import learning_orchestrator
        results = learning_orchestrator.run_complete_learning_cycle()
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning cycle failed: {str(e)}")


@app.post("/api/learning/recognize-patterns")
async def recognize_feedback_patterns():
    """Recognize patterns in user feedback"""
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from pattern_recognizer import analyze_feedback_patterns
        patterns = analyze_feedback_patterns()
        return {"success": True, "patterns": patterns, "count": len(patterns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern recognition failed: {str(e)}")


@app.post("/api/learning/tune-algorithm")
async def tune_algorithm():
    """Auto-tune algorithm based on feedback patterns"""
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from algorithm_tuner import auto_tune_algorithm
        results = auto_tune_algorithm()
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Algorithm tuning failed: {str(e)}")


@app.get("/api/learning/system-status")
async def get_learning_system_status():
    """Get comprehensive learning system status"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False}
    
    try:
        from learning_orchestrator import learning_orchestrator
        status = learning_orchestrator.get_learning_system_status()
        return {"enabled": True, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@app.get("/api/learning/algorithm-config")
async def get_algorithm_config():
    """Get current algorithm configuration"""
    if not Config.LEARNING_ENABLED:
        return {"enabled": False}
    
    try:
        from algorithm_tuner import algorithm_tuner
        config = algorithm_tuner.get_current_algorithm_config()
        return {"enabled": True, "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm config: {str(e)}")


@app.post("/api/learning/smart-learning")
async def run_smart_learning():
    """Run smart learning that decides what to do based on available data"""
    if not Config.LEARNING_ENABLED:
        return {"success": False, "message": "Learning system is disabled"}
    
    try:
        from learning_orchestrator import run_smart_learning
        results = run_smart_learning()
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart learning failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

