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

# Import position tracking
from position_tracker import fix_entry_ordering_enhanced_method_1

# Import real-time processing pipeline
from realtime_pipeline import execute_realtime_pipeline

# Context Memory System for LLM
class LLMContextManager:
    """
    Enhanced context manager for LLM with robust parallel processing support
    Manages conversation context and provides better error handling and progress tracking
    """
    def __init__(self, max_context=5):
        self.conversation_history = []
        self.max_context = max_context
        self.processed_chunks = 0
        self.failed_chunks = 0
        self.context_lock = asyncio.Lock()  # Thread-safe context updates
        self.field_patterns = {}  # Track field naming patterns for consistency
        self.processing_stats = {
            "total_entries": 0,
            "avg_entries_per_chunk": 0,
            "common_fields": set(),
            "processing_errors": []
        }
    
    async def add_interaction(self, chunk_content: str, llm_response: str, chunk_id: int = None, entries_count: int = 0):
        """Add a chunk processing interaction to context with thread safety"""
        async with self.context_lock:
            try:
                # Validate inputs
                if not chunk_content or not llm_response:
                    print(f"⚠️  Warning: Empty content or response for chunk {chunk_id}")
                    return
                
                # Truncate content for context efficiency
                truncated_content = chunk_content[:500] + "..." if len(chunk_content) > 500 else chunk_content
                truncated_response = llm_response[:800] + "..." if len(llm_response) > 800 else llm_response
                
                interaction = {
                    "role": "user", 
                    "content": f"PREVIOUS CHUNK {chunk_id or self.processed_chunks + 1}:\n{truncated_content}"
                }
                response = {
                    "role": "assistant",
                    "content": truncated_response
                }
                
                self.conversation_history.extend([interaction, response])
                self.processed_chunks += 1
                
                # Update processing stats
                self.processing_stats["total_entries"] += entries_count
                if self.processed_chunks > 0:
                    self.processing_stats["avg_entries_per_chunk"] = self.processing_stats["total_entries"] / self.processed_chunks
                
                # Extract field patterns for consistency
                self._extract_field_patterns(llm_response)
                
                # Keep only last N interactions (each interaction = user + assistant)
                max_messages = self.max_context * 2
                if len(self.conversation_history) > max_messages:
                    self.conversation_history = self.conversation_history[-max_messages:]
                    
            except Exception as e:
                print(f"❌ Error adding interaction to context: {e}")
                self.processing_stats["processing_errors"].append(f"Context update error: {str(e)}")
    
    def _extract_field_patterns(self, llm_response: str):
        """Extract field naming patterns from LLM response for consistency"""
        try:
            import json
            # Try to parse JSON and extract field names
            if '"Key"' in llm_response and '"Value"' in llm_response:
                # Look for key patterns in the response
                import re
                key_matches = re.findall(r'"Key":\s*"([^"]+)"', llm_response)
                for key in key_matches:
                    if key:
                        # Normalize key for pattern matching
                        normalized_key = key.lower().strip()
                        if normalized_key not in self.field_patterns:
                            self.field_patterns[normalized_key] = key
                        self.processing_stats["common_fields"].add(key)
        except Exception as e:
            # Silently handle pattern extraction errors
            pass
    
    async def get_context_messages(self, current_chunk: str, chunk_id: int = None) -> list:
        """Get messages with enhanced context for current chunk"""
        async with self.context_lock:
            messages = []
            
            # Add conversation history for context
            messages.extend(self.conversation_history)
            
            # Add current chunk with enhanced context-aware prompt
            from prompt_templates import get_expected_format_prompt
            base_prompt = get_expected_format_prompt(current_chunk)
            
            # Build consistency guidance from previous patterns
            consistency_guidance = ""
            if self.field_patterns:
                common_fields = list(self.processing_stats["common_fields"])[:10]  # Top 10 most common
                if common_fields:
                    consistency_guidance = f"\nCOMMON FIELD NAMES from previous chunks: {', '.join(common_fields)}\nUse these exact field names when applicable for consistency."
            
            context_prompt = f"""
PROCESSING CONTEXT: You have successfully processed {self.processed_chunks} previous chunks from this document.
Average entries per chunk: {self.processing_stats["avg_entries_per_chunk"]:.1f}

Maintain consistency with previous extractions in terms of:
- Field naming conventions (use exact same field names as shown in previous responses)
- Data formatting patterns (same date/number formats)
- Value extraction style (same level of detail)
- JSON structure (same format as previous responses)
{consistency_guidance}

Look at the previous chunk responses above for reference patterns.

{base_prompt}
"""
            
            messages.append({"role": "user", "content": context_prompt})
            return messages
    
    async def record_chunk_error(self, chunk_id: int, error: str):
        """Record a chunk processing error"""
        async with self.context_lock:
            self.failed_chunks += 1
            self.processing_stats["processing_errors"].append(f"Chunk {chunk_id}: {error}")
    
    def clear_context(self):
        """Clear conversation history (for new document)"""
        self.conversation_history = []
        self.processed_chunks = 0
        self.failed_chunks = 0
        self.field_patterns = {}
        self.processing_stats = {
            "total_entries": 0,
            "avg_entries_per_chunk": 0,
            "common_fields": set(),
            "processing_errors": []
        }
    
    def get_context_summary(self) -> dict:
        """Get enhanced summary of current context state"""
        return {
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "context_messages": len(self.conversation_history),
            "max_context": self.max_context,
            "total_entries": self.processing_stats["total_entries"],
            "avg_entries_per_chunk": round(self.processing_stats["avg_entries_per_chunk"], 1),
            "common_fields_count": len(self.processing_stats["common_fields"]),
            "processing_errors_count": len(self.processing_stats["processing_errors"]),
            "success_rate": round((self.processed_chunks / max(1, self.processed_chunks + self.failed_chunks)) * 100, 1)
        }

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
        target_sections = min(8, max(1, total_lines // 50))  # 1-8 sections (min 1 to avoid division by zero)
        section_size = max(1, total_lines // target_sections)  # Ensure section_size is at least 1
        
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
    
    # Fallback: if no chunks were created (text too short), create one chunk with all content
    if not chunks and full_text.strip():
        chunks.append({
            'chunk_id': 1,
            'content': full_text.strip(),
            'header': 'Full Content',
            'section_type': 'full_content',
            'start_line': 0,
            'end_line': len(lines),
            'length': len(full_text),
            'start_page': 1,
            'end_page': total_pages,
            'page_range': '1'
        })
    
    return chunks

# Context-Aware Chunk Processing
async def process_chunk_with_context(chunk: Dict[str, Any], context_manager: LLMContextManager) -> Dict[str, Any]:
    """
    Enhanced chunk processing with robust error handling and null value protection
    
    Args:
        chunk: Chunk data with content
        context_manager: Enhanced context manager with conversation history
    
    Returns:
        Processed chunk result with entries and metadata
    """
    chunk_id = chunk.get('chunk_id', 0)
    
    try:
        # Input validation and null value protection
        if not chunk or not isinstance(chunk, dict):
            raise ValueError("Invalid chunk data: chunk must be a non-empty dictionary")
        
        chunk_content = chunk.get('content', '').strip()
        if not chunk_content:
            raise ValueError("Invalid chunk content: content cannot be empty")
        
        if len(chunk_content) < 10:
            raise ValueError(f"Invalid chunk content: content too short ({len(chunk_content)} chars)")
        
        from openai import AsyncOpenAI
        import os
        
        # Validate API credentials
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        try:
            client = AsyncOpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize AsyncOpenAI client: {str(e)}")
        
        # Get context-aware messages with error handling
        try:
            if context_manager.processed_chunks == 0:
                # First chunk - use standard prompt
                from prompt_templates import get_expected_format_prompt
                prompt = get_expected_format_prompt(chunk_content)
                messages = [{"role": "user", "content": prompt}]
            else:
                # Subsequent chunks - use context-aware messages
                messages = await context_manager.get_context_messages(chunk_content, chunk_id)
        except Exception as e:
            print(f"⚠️  Context message generation failed for chunk {chunk_id}, using fallback: {e}")
            # Fallback to basic prompt
            from prompt_templates import get_expected_format_prompt
            prompt = get_expected_format_prompt(chunk_content)
            messages = [{"role": "user", "content": prompt}]
        
        # Validate messages before API call
        if not messages or not isinstance(messages, list):
            raise ValueError("Invalid messages format for LLM API call")
        
        # Process with LLM with timeout protection - using async client for true parallel processing
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                timeout=60  # 60 second timeout
            )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")
        
        # Validate API response
        if not response or not response.choices:
            raise ValueError("Invalid API response: no choices returned")
        
        raw_llm_response = response.choices[0].message.content
        if not raw_llm_response:
            raise ValueError("Invalid API response: empty content")
        
        # Parse response with enhanced error handling
        try:
            parsed_data = clean_and_parse_json(raw_llm_response)
            if not isinstance(parsed_data, dict):
                raise ValueError("Parsed data is not a dictionary")
        except Exception as e:
            print(f"⚠️  JSON parsing failed for chunk {chunk_id}: {e}")
            # Try to extract entries manually as fallback
            parsed_data = {"entries": [], "global_notes": None}
        
        chunk_entries = parsed_data.get('entries', [])
        
        # Validate and clean entries
        if not isinstance(chunk_entries, list):
            print(f"⚠️  Invalid entries format for chunk {chunk_id}, using empty list")
            chunk_entries = []
        
        # Clean and validate each entry
        cleaned_entries = []
        for i, entry in enumerate(chunk_entries):
            try:
                if not isinstance(entry, dict):
                    print(f"⚠️  Skipping invalid entry {i} in chunk {chunk_id}: not a dictionary")
                    continue
                
                # Ensure required fields exist and are not null
                key = entry.get('Key', '').strip() if entry.get('Key') else ''
                value = entry.get('Value', '').strip() if entry.get('Value') else ''
                
                if not key or not value:
                    print(f"⚠️  Skipping entry {i} in chunk {chunk_id}: missing key or value")
                    continue
                
                # Clean entry
                cleaned_entry = {
                    'Key': key,
                    'Value': value,
                    'Comments': entry.get('Comments', '').strip() if entry.get('Comments') else '',
                    'chunk_id': chunk_id
                }
                
                cleaned_entries.append(cleaned_entry)
                
            except Exception as e:
                print(f"⚠️  Error processing entry {i} in chunk {chunk_id}: {e}")
                continue
        
        # Update context with successful processing
        try:
            await context_manager.add_interaction(
                chunk_content, 
                raw_llm_response, 
                chunk_id, 
                len(cleaned_entries)
            )
        except Exception as e:
            print(f"⚠️  Failed to update context for chunk {chunk_id}: {e}")
        
        return {
            "chunk_id": chunk_id,
            "entries": cleaned_entries,
            "global_notes": parsed_data.get('global_notes'),
            "raw_response": raw_llm_response,
            "context_used": len(context_manager.conversation_history) > 0,
            "processing_time": None,  # Will be set by retry wrapper
            "entries_count": len(cleaned_entries),
            "status": "success"
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error processing chunk {chunk_id}: {error_msg}")
        
        # Record error in context manager
        try:
            await context_manager.record_chunk_error(chunk_id, error_msg)
        except Exception as context_error:
            print(f"⚠️  Failed to record error in context: {context_error}")
        
        return {
            "chunk_id": chunk_id,
            "entries": [],
            "global_notes": None,
            "error": error_msg,
            "context_used": False,
            "processing_time": None,
            "entries_count": 0,
            "status": "error"
        }


# Batch Parallel Processing with Rate Limit Protection
async def process_chunk_with_retry(chunk: Dict[str, Any], context_manager: LLMContextManager, semaphore: asyncio.Semaphore, max_retries: int = 3) -> Dict[str, Any]:
    """
    Enhanced chunk processing with comprehensive retry logic, error isolation, and timing
    
    Args:
        chunk: Chunk data to process
        context_manager: Enhanced context manager for conversation history
        semaphore: Semaphore to limit concurrent requests
        max_retries: Maximum number of retry attempts
    
    Returns:
        Processed chunk result with enhanced error handling and timing data
    """
    import time
    
    chunk_id = chunk.get('chunk_id', 0)
    start_time = time.time()
    
    async with semaphore:  # Limit concurrent requests
        last_error = None
        
        for attempt in range(max_retries):
            attempt_start = time.time()
            
            try:
                # Process chunk with context
                result = await process_chunk_with_context(chunk, context_manager)
                
                # Add timing information
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                result["retry_attempts"] = attempt + 1
                
                if attempt > 0:
                    print(f"✅ Chunk {chunk_id} succeeded on attempt {attempt + 1} after {processing_time:.1f}s")
                
                return result
            
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                attempt_duration = time.time() - attempt_start
                
                print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed for chunk {chunk_id} after {attempt_duration:.1f}s: {str(e)[:100]}")
                
                # Categorize error types for better handling
                if "rate limit" in error_str or "429" in error_str:
                    # Rate limit errors - exponential backoff
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff with jitter
                        print(f"⏳ Rate limit hit for chunk {chunk_id}, waiting {wait_time:.1f}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                
                elif "timeout" in error_str or "timed out" in error_str:
                    # Timeout errors - shorter backoff
                    if attempt < max_retries - 1:
                        wait_time = 2 + attempt
                        print(f"⏳ Timeout for chunk {chunk_id}, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                
                elif "api" in error_str or "openai" in error_str or "connection" in error_str:
                    # API/Connection errors - moderate backoff
                    if attempt < max_retries - 1:
                        wait_time = 1 + (attempt * 0.5)
                        print(f"⏳ API error for chunk {chunk_id}, waiting {wait_time:.1f}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                
                elif "invalid" in error_str or "parse" in error_str or "json" in error_str:
                    # Data/parsing errors - immediate retry (no wait)
                    if attempt < max_retries - 1:
                        print(f"🔄 Data error for chunk {chunk_id}, retrying immediately...")
                        continue
                
                else:
                    # Unknown errors - short wait
                    if attempt < max_retries - 1:
                        wait_time = 0.5 + attempt
                        print(f"⏳ Unknown error for chunk {chunk_id}, waiting {wait_time:.1f}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
        
        # All retries exhausted
        total_time = time.time() - start_time
        final_error = str(last_error) if last_error else "Unknown error"
        
        print(f"❌ Chunk {chunk_id} failed after {max_retries} attempts in {total_time:.1f}s: {final_error[:100]}")
        
        # Record final failure in context manager
        try:
            await context_manager.record_chunk_error(chunk_id, final_error)
        except Exception as context_error:
            print(f"⚠️  Failed to record final error in context: {context_error}")
        
        return {
            "chunk_id": chunk_id,
            "entries": [],
            "global_notes": None,
            "error": final_error,
            "context_used": False,
            "processing_time": total_time,
            "retry_attempts": max_retries,
            "entries_count": 0,
            "status": "error",
            "error_category": _categorize_error(final_error)
        }


def _categorize_error(error_str: str) -> str:
    """Categorize error for better reporting and handling"""
    error_lower = error_str.lower()
    
    if "rate limit" in error_lower or "429" in error_lower:
        return "rate_limit"
    elif "timeout" in error_lower or "timed out" in error_lower:
        return "timeout"
    elif "api" in error_lower or "openai" in error_lower:
        return "api_error"
    elif "connection" in error_lower or "network" in error_lower:
        return "connection_error"
    elif "invalid" in error_lower or "parse" in error_lower or "json" in error_lower:
        return "data_error"
    elif "auth" in error_lower or "key" in error_lower:
        return "auth_error"
    else:
        return "unknown_error"


async def process_chunks_batch_parallel(chunks: List[Dict[str, Any]], context_manager: LLMContextManager, batch_size: int = 5, progress_callback=None, return_progress_events: bool = False) -> List[Dict[str, Any]]:
    """
    Enhanced parallel batch processing with robust error handling, context memory preservation, and detailed progress reporting
    
    Args:
        chunks: List of chunks to process
        context_manager: Enhanced context manager for conversation history
        batch_size: Number of chunks to process concurrently
        progress_callback: Optional callback function for progress updates
    
    Returns:
        List of processed chunk results with enhanced metadata
    """
    import time
    
    if not chunks:
        print("⚠️  No chunks to process")
        return []
    
    # Validate inputs
    if not isinstance(chunks, list):
        raise ValueError("Chunks must be a list")
    
    if batch_size <= 0:
        batch_size = 5
        print(f"⚠️  Invalid batch_size, using default: {batch_size}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(batch_size)
    
    # Initialize processing metrics
    start_time = time.time()
    processing_stats = {
        "total_chunks": len(chunks),
        "batch_size": batch_size,
        "successful_chunks": 0,
        "failed_chunks": 0,
        "total_entries": 0,
        "processing_errors": [],
        "batch_timings": [],
        "context_preservation_rate": 0
    }
    
    # Store progress events if requested
    progress_events = [] if return_progress_events else None
    
    print(f"🚀 Starting enhanced batch parallel processing:")
    print(f"   📊 Total chunks: {len(chunks)}")
    print(f"   🔄 Batch size: {batch_size}")
    print(f"   🧠 Context manager: {context_manager.get_context_summary()}")
    
    # Create tasks for all chunks with enhanced retry logic
    tasks = []
    for chunk in chunks:
        task = process_chunk_with_retry(chunk, context_manager, semaphore, max_retries=3)
        tasks.append(task)
    
    # Execute tasks in batches with detailed progress tracking
    results = []
    completed = 0
    
    # Process in batches for better progress tracking and error isolation
    for batch_idx in range(0, len(tasks), batch_size):
        batch_start_time = time.time()
        batch_tasks = tasks[batch_idx:batch_idx+batch_size]
        batch_chunk_ids = [chunks[batch_idx + i].get('chunk_id', batch_idx + i + 1) for i in range(len(batch_tasks))]
        
        print(f"🔄 Processing batch {batch_idx//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}: chunks {batch_chunk_ids}")
        
        # Send chunk_start events for each chunk in this batch
        if return_progress_events and progress_events is not None:
            for j in range(len(batch_tasks)):
                chunk_id = chunks[batch_idx + j].get('chunk_id', batch_idx + j + 1)
                chunk_info = chunks[batch_idx + j]
                
                progress_events.append({
                    'event': 'chunk_start',
                    'chunk': chunk_id,
                    'total_chunks': len(chunks),
                    'page_range': chunk_info.get('page_range', ''),
                    'section': chunk_info.get('section_type', ''),
                    'header': chunk_info.get('header', ''),
                    'char_range': chunk_info.get('char_range', ''),
                    'status': 'started'
                })
        
        try:
            # Execute batch with timeout protection
            batch_results = await asyncio.wait_for(
                asyncio.gather(*batch_tasks, return_exceptions=True),
                timeout=300  # 5 minute timeout per batch
            )
        except asyncio.TimeoutError:
            print(f"⏰ Batch {batch_idx//batch_size + 1} timed out, creating error results")
            batch_results = [Exception("Batch processing timeout") for _ in batch_tasks]
        except Exception as e:
            print(f"❌ Batch {batch_idx//batch_size + 1} failed with exception: {e}")
            batch_results = [Exception(f"Batch processing error: {str(e)}") for _ in batch_tasks]
        
        # Process batch results with error isolation
        batch_successful = 0
        batch_failed = 0
        batch_entries = 0
        
        for j, result in enumerate(batch_results):
            chunk_id = chunks[batch_idx + j].get('chunk_id', batch_idx + j + 1)
            
            if isinstance(result, Exception):
                error_msg = str(result)
                print(f"❌ Exception in chunk {chunk_id}: {error_msg}")
                
                # Record error in processing stats
                processing_stats["processing_errors"].append(f"Chunk {chunk_id}: {error_msg}")
                
                # Create error result
                error_result = {
                    "chunk_id": chunk_id,
                    "entries": [],
                    "global_notes": None,
                    "error": error_msg,
                    "context_used": False,
                    "processing_time": None,
                    "entries_count": 0,
                    "status": "error",
                    "batch_id": batch_idx//batch_size + 1
                }
                results.append(error_result)
                batch_failed += 1
                processing_stats["failed_chunks"] += 1
                
                # Record error in context manager
                try:
                    await context_manager.record_chunk_error(chunk_id, error_msg)
                except Exception as context_error:
                    print(f"⚠️  Failed to record error in context: {context_error}")
                
            else:
                # Successful result
                if result and isinstance(result, dict):
                    result["batch_id"] = batch_idx//batch_size + 1
                    entries_count = len(result.get('entries', []))
                    batch_entries += entries_count
                    processing_stats["total_entries"] += entries_count
                    
                    if result.get('status') == 'success':
                        batch_successful += 1
                        processing_stats["successful_chunks"] += 1
                    else:
                        batch_failed += 1
                        processing_stats["failed_chunks"] += 1
                        if result.get('error'):
                            processing_stats["processing_errors"].append(f"Chunk {chunk_id}: {result.get('error')}")
                
                results.append(result)
        
        # Record batch timing
        batch_duration = time.time() - batch_start_time
        processing_stats["batch_timings"].append({
            "batch_id": batch_idx//batch_size + 1,
            "duration": batch_duration,
            "chunks_processed": len(batch_tasks),
            "successful": batch_successful,
            "failed": batch_failed,
            "entries": batch_entries
        })
        
        completed += len(batch_tasks)
        progress = (completed / len(chunks)) * 100
        
        # Detailed progress reporting
        print(f"📊 Batch {batch_idx//batch_size + 1} complete:")
        print(f"   ✅ Successful: {batch_successful}/{len(batch_tasks)}")
        print(f"   ❌ Failed: {batch_failed}/{len(batch_tasks)}")
        print(f"   📝 Entries: {batch_entries}")
        print(f"   ⏱️  Duration: {batch_duration:.1f}s")
        print(f"   📈 Overall progress: {completed}/{len(chunks)} chunks ({progress:.1f}%)")
        
        # Always provide progress information for SSE streaming
        progress_data = {
            "completed": completed,
            "total": len(chunks),
            "progress": progress,
            "batch_id": batch_idx//batch_size + 1,
            "batch_successful": batch_successful,
            "batch_failed": batch_failed,
            "batch_entries": batch_entries,
            "context_summary": context_manager.get_context_summary()
        }
        
        # Call progress callback if provided
        if progress_callback:
            try:
                await progress_callback(progress_data)
            except Exception as e:
                print(f"⚠️  Progress callback failed: {e}")
        
        # Store individual chunk progress events if requested (format for frontend compatibility)
        if return_progress_events and progress_events is not None:
            # Emit individual chunk events for each chunk in this batch
            for j, result in enumerate(batch_results):
                chunk_id = chunks[batch_idx + j].get('chunk_id', batch_idx + j + 1)
                
                if isinstance(result, Exception):
                    # Chunk failed
                    progress_events.append({
                        'event': 'chunk_complete',
                        'chunk': chunk_id,
                        'total_chunks': len(chunks),
                        'status': 'error',
                        'error': str(result),
                        'entries_count': 0,
                        'processing_time': None
                    })
                else:
                    # Chunk succeeded
                    progress_events.append({
                        'event': 'chunk_complete',
                        'chunk': chunk_id,
                        'total_chunks': len(chunks),
                        'status': result.get('status', 'success'),
                        'entries_count': result.get('entries_count', 0),
                        'processing_time': result.get('processing_time')
                    })
    
    # Calculate final statistics
    total_duration = time.time() - start_time
    context_summary = context_manager.get_context_summary()
    processing_stats["context_preservation_rate"] = context_summary.get("success_rate", 0)
    
    # Final summary
    print(f"\n✅ Enhanced batch processing complete:")
    print(f"   📊 Total processed: {len(results)} chunks")
    print(f"   ✅ Successful: {processing_stats['successful_chunks']}")
    print(f"   ❌ Failed: {processing_stats['failed_chunks']}")
    print(f"   📝 Total entries: {processing_stats['total_entries']}")
    print(f"   ⏱️  Total duration: {total_duration:.1f}s")
    print(f"   🧠 Context preservation rate: {processing_stats['context_preservation_rate']:.1f}%")
    print(f"   📈 Average entries per chunk: {context_summary.get('avg_entries_per_chunk', 0):.1f}")
    
    if processing_stats["processing_errors"]:
        print(f"   ⚠️  Processing errors: {len(processing_stats['processing_errors'])}")
        for error in processing_stats["processing_errors"][:5]:  # Show first 5 errors
            print(f"      - {error}")
        if len(processing_stats["processing_errors"]) > 5:
            print(f"      ... and {len(processing_stats['processing_errors']) - 5} more errors")
    
    # Return results with optional progress events
    if return_progress_events:
        return results, progress_events
    else:
        return results


# Initialize FastAPI app
app = FastAPI(title="AI Document Processor API")

@app.get("/")
async def root():
    """Root endpoint for basic connectivity test"""
    return {"message": "AI Document Processor API", "status": "running"}

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel deployments
    allow_origins=[
        "http://localhost:3001",
        "http://localhost:3000",
    ],
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

# ============================================================================
# UNIFIED PROCESSING PIPELINE
# ============================================================================

async def unified_processing_pipeline(
    full_text: str, 
    total_pages: int, 
    input_type: str = "unknown",
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """
    Unified processing pipeline for all input types (PDF upload, text input, file path)
    Uses consistent parallel processing with AsyncOpenAI and real-time SSE progress
    
    Args:
        full_text: Extracted text content to process
        total_pages: Number of pages (for PDF) or 1 for text input
        input_type: Type of input ("pdf_upload", "text_input", "file_path")
        session_id: Optional session ID for tracking
    
    Yields:
        SSE formatted JSON events for real-time progress updates
    """
    try:
        start_time = time.time()
        
        # Initialize performance tracker
        tracker = get_performance_tracker()
        if not session_id:
            session_id = tracker.create_session_id()
        
        # Phase 1: Text Analysis & Semantic Chunking
        yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'analyzing_content', 'input_type': input_type, 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Create semantic chunks using intelligent chunking
        semantic_chunks = create_intelligent_chunks(full_text, total_pages)
        total_chunks = len(semantic_chunks)
        
        yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'semantic_chunking', 'input_type': input_type, 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Phase 2: Initialize Context Memory
        context_manager = LLMContextManager(max_context=5)
        
        yield f"data: {json.dumps({'event': 'context_initialized', 'max_context': 5, 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Phase 3: PARALLEL Chunk Processing with Real-time Progress Events
        yield f"data: {json.dumps({'event': 'batch_processing_start', 'strategy': 'parallel_with_realtime_events', 'concurrency': min(3, total_chunks), 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Send initial "started" events for all chunks
        for i, chunk in enumerate(semantic_chunks):
            chunk_num = i + 1
            yield f"data: {json.dumps({'event': 'chunk_start', 'chunk': chunk_num, 'total_chunks': total_chunks, 'page_range': chunk.get('page_range', '1'), 'section': chunk.get('header', f'Section {chunk_num}'), 'status': 'started', 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Create async wrapper that tracks timing and chunk_id
        async def process_chunk_with_timing(chunk, chunk_id, semaphore):
            async with semaphore:
                start_time = time.time()
                result = await process_chunk_with_context(chunk, context_manager)
                result['processing_time'] = round(time.time() - start_time, 2)
                result['chunk_id'] = chunk_id
                return result
        
        # Process all chunks in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent LLM calls
        tasks = []
        for i, chunk in enumerate(semantic_chunks):
            task = asyncio.create_task(process_chunk_with_timing(chunk, i + 1, semaphore))
            tasks.append(task)
        
        # Collect results as they complete and emit SSE events
        chunk_results = [None] * total_chunks  # Pre-allocate to maintain order
        for completed_task in asyncio.as_completed(tasks):
            chunk_result = await completed_task
            chunk_num = chunk_result.get('chunk_id', 0)
            chunk_results[chunk_num - 1] = chunk_result
            
            # Send completion event immediately when chunk finishes
            status = 'success' if not chunk_result.get('error') else 'error'
            yield f"data: {json.dumps({'event': 'chunk_complete', 'chunk': chunk_num, 'total_chunks': total_chunks, 'status': status, 'entries_count': chunk_result.get('entries_count', 0), 'processing_time': chunk_result.get('processing_time'), 'error': chunk_result.get('error'), 'development_mode': True})}\n\n"
            await asyncio.sleep(0.05)  # Small delay to ensure event is flushed
        
        # Send batch processing complete event
        successful_chunks = sum(1 for r in chunk_results if not r.get('error'))
        failed_chunks = len(chunk_results) - successful_chunks
        
        yield f"data: {json.dumps({'event': 'batch_processing_complete', 'successful': successful_chunks, 'failed': failed_chunks, 'context_summary': context_manager.get_context_summary(), 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Phase 4: Enhanced Deduplication with Fuzzy Matching
        yield f"data: {json.dumps({'event': 'merging_start', 'status': 'deduplicating', 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Collect all entries from chunks
        all_entries = []
        global_notes = None
        
        for chunk_result in chunk_results:
            if not chunk_result.get('error'):
                entries = chunk_result.get('entries', [])
                all_entries.extend(entries)
                if not global_notes and chunk_result.get('global_notes'):
                    global_notes = chunk_result.get('global_notes')
        
        initial_count = len(all_entries)
        
        # Run full deduplication pipeline
        yield f"data: {json.dumps({'event': 'dedup_progress', 'status': 'running_deduplication', 'initial_count': initial_count, 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        deduplicated_entries, dedup_stats = full_deduplication_pipeline(
            all_entries,
            fuzzy_threshold=Config.FUZZY_MATCH_THRESHOLD,
            enable_fuzzy=Config.ENABLE_FUZZY_DEDUP,
            enable_repair=Config.ENABLE_TRUNCATION_REPAIR
        )
        
        final_count = len(deduplicated_entries)
        duplicates_removed = initial_count - final_count
        
        yield f"data: {json.dumps({'event': 'dedup_complete', 'initial_count': initial_count, 'final_count': final_count, 'duplicates_removed': duplicates_removed, 'dedup_stats': dedup_stats, 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Phase 5: Final Processing & Results
        processing_time = time.time() - start_time
        
        # Create chunk info for frontend
        chunks_info = []
        for i, (chunk, result) in enumerate(zip(semantic_chunks, chunk_results)):
            chunk_info = {
                'chunk_id': i + 1,
                'page_range': chunk.get('page_range', '1'),
                'section_type': chunk.get('section_type', 'content'),
                'header': chunk.get('header', f'Section {i + 1}'),
                'entries_count': result.get('entries_count', 0) if result else 0,
                'processing_time': result.get('processing_time', 0) if result else 0,
                'status': result.get('status', 'error') if result else 'error',
                'error': result.get('error') if result and result.get('error') else None
            }
            chunks_info.append(chunk_info)
        
        # Send final completion event
        final_result = {
            'event': 'enhanced_complete',
            'entries': deduplicated_entries,
            'global_notes': global_notes,
            'session_id': session_id,
            'chunks': chunks_info,
            'chunk_results': [{'chunk_id': i + 1, 'entries': result.get('entries', []) if result else []} for i, result in enumerate(chunk_results)],
            'total_pages': total_pages,
            'total_chunks': total_chunks,
            'processing_time': processing_time,
            'chunk_size': Config.CHUNK_SIZE,
            'input_type': input_type,
            'dedup_stats': dedup_stats,
            'development_mode': True
        }
        
        yield f"data: {json.dumps(final_result)}\n\n"
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in unified processing pipeline: {error_msg}")
        yield f"data: {json.dumps({'event': 'error', 'message': error_msg, 'input_type': input_type})}\n\n"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Document Processor API is running"}


# ============================================================================
# UNIFIED PROCESSING PIPELINE
# ============================================================================

async def unified_processing_pipeline(
    full_text: str,
    total_pages: int,
    input_type: str = "unknown",
    session_id: str = None
) -> AsyncGenerator[str, None]:
    """
    Unified processing pipeline for all input types (PDF upload, text input, file path)
    Uses TRUE PARALLEL PROCESSING with AsyncOpenAI for maximum performance
    
    Args:
        full_text: Extracted text content to process
        total_pages: Number of pages (1 for text input)
        input_type: Type of input ("pdf_upload", "text_input", "file_path")
        session_id: Optional session ID for tracking
    
    Yields:
        SSE events as JSON strings
    """
    try:
        start_time = time.time()
        
        # Phase 1: Semantic Analysis & Chunking
        yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'analyzing_content', 'input_type': input_type})}\n\n"
        await asyncio.sleep(0.1)
        
        # Create semantic chunks (same for all input types)
        semantic_chunks = create_intelligent_chunks(full_text, total_pages)
        total_chunks = len(semantic_chunks)
        
        yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'semantic_chunking', 'total_pages': total_pages})}\n\n"
        await asyncio.sleep(0.1)
        
        # Phase 2: Initialize Context Memory
        context_manager = LLMContextManager(max_context=5)
        
        yield f"data: {json.dumps({'event': 'context_initialized', 'max_context': 5})}\n\n"
        await asyncio.sleep(0.1)
        
        # Phase 3: TRUE PARALLEL Processing with Real-time Progress
        yield f"data: {json.dumps({'event': 'batch_processing_start', 'strategy': 'true_parallel', 'concurrency': min(3, total_chunks)})}\n\n"
        await asyncio.sleep(0.1)
        
        # Send initial "started" events for all chunks
        for i, chunk in enumerate(semantic_chunks):
            chunk_num = i + 1
            yield f"data: {json.dumps({'event': 'chunk_start', 'chunk': chunk_num, 'total_chunks': total_chunks, 'page_range': chunk.get('page_range', '1'), 'section': chunk.get('header', f'Section {chunk_num}'), 'status': 'started'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Create async wrapper that tracks timing and chunk_id
        async def process_chunk_with_timing(chunk, chunk_id, semaphore):
            async with semaphore:
                start_time = time.time()
                result = await process_chunk_with_context(chunk, context_manager)
                result['processing_time'] = round(time.time() - start_time, 2)
                result['chunk_id'] = chunk_id
                return result
        
        # Process all chunks in TRUE PARALLEL with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent LLM calls
        tasks = []
        for i, chunk in enumerate(semantic_chunks):
            task = asyncio.create_task(process_chunk_with_timing(chunk, i + 1, semaphore))
            tasks.append(task)
        
        # Collect results as they complete (TRUE PARALLEL - out of order completion)
        chunk_results = [None] * total_chunks  # Pre-allocate to maintain order
        for completed_task in asyncio.as_completed(tasks):
            chunk_result = await completed_task
            chunk_num = chunk_result.get('chunk_id', 0)
            chunk_results[chunk_num - 1] = chunk_result
            
            # Send completion event immediately when chunk finishes
            status = 'success' if not chunk_result.get('error') else 'error'
            yield f"data: {json.dumps({'event': 'chunk_complete', 'chunk': chunk_num, 'total_chunks': total_chunks, 'status': status, 'entries_count': chunk_result.get('entries_count', 0), 'processing_time': chunk_result.get('processing_time'), 'error': chunk_result.get('error')})}\n\n"
            await asyncio.sleep(0.05)
        
        # Send batch processing complete event
        successful_chunks = sum(1 for r in chunk_results if not r.get('error'))
        failed_chunks = len(chunk_results) - successful_chunks
        
        yield f"data: {json.dumps({'event': 'batch_processing_complete', 'successful': successful_chunks, 'failed': failed_chunks, 'context_summary': context_manager.get_context_summary()})}\n\n"
        await asyncio.sleep(0.1)
        
        # Phase 4: Deduplication
        yield f"data: {json.dumps({'event': 'merging_start', 'status': 'deduplicating'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Collect all entries from chunks
        all_entries = []
        global_notes = None
        
        for chunk_result in chunk_results:
            if not chunk_result.get('error'):
                entries = chunk_result.get('entries', [])
                all_entries.extend(entries)
                if not global_notes and chunk_result.get('global_notes'):
                    global_notes = chunk_result.get('global_notes')
        
        initial_count = len(all_entries)
        
        # Run full deduplication pipeline
        deduplicated_entries, dedup_stats = full_deduplication_pipeline(
            all_entries,
            fuzzy_threshold=Config.FUZZY_MATCH_THRESHOLD,
            enable_fuzzy=Config.ENABLE_FUZZY_DEDUP,
            enable_repair=Config.ENABLE_TRUNCATION_REPAIR
        )
        
        final_count = len(deduplicated_entries)
        duplicates_removed = initial_count - final_count
        
        yield f"data: {json.dumps({'event': 'dedup_complete', 'initial_count': initial_count, 'final_count': final_count, 'duplicates_removed': duplicates_removed, 'dedup_stats': dedup_stats, 'development_mode': True})}\n\n"
        await asyncio.sleep(0.1)
        
        # Continue with position tracking and final result...
        # (This function appears to be incomplete - should be completed or removed)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in unified processing pipeline: {error_msg}")
        yield f"data: {json.dumps({'event': 'error', 'message': error_msg, 'input_type': input_type})}\n\n"


@app.post("/api/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    PDF upload endpoint - redirects to enhanced processing
    """
    return await process_pdf_enhanced(file)

@app.post("/api/process-enhanced")
async def process_pdf_enhanced(file: UploadFile = File(...)):
    """
    Enhanced PDF processing using unified pipeline for consistent real-time progress
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
            
            # Phase 1: Text Extraction & Analysis (identical to other endpoints)
            yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'extracting_text', 'development_mode': True, 'input_type': 'pdf_upload'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Extract full text and get page count
            full_text = extract_text_from_pdf(temp_pdf_path)
            reader = PdfReader(temp_pdf_path)
            total_pages = len(reader.pages)
            
            semantic_chunks = create_intelligent_chunks(full_text, total_pages)
            total_chunks = len(semantic_chunks)
            
            yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'semantic_chunking', 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            # Use real-time processing pipeline for immediate progress updates
            context_manager = LLMContextManager(max_context=5)
            async for event in execute_realtime_pipeline(
                semantic_chunks, full_text, total_pages, session_id,
                file.filename, len(content), start_time, "pdf_upload",
                context_manager, process_chunk_with_context
            ):
                yield event
                
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'development_mode': True, 'input_type': 'pdf_upload'})}\n\n"
        
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


@app.post("/api/process-text")
async def process_text(data: Dict[str, str]):
    """
    Text input endpoint - uses unified processing
    """
    text_content = data.get('text', '').strip()
    if not text_content:
        raise HTTPException(status_code=400, detail="Text content is required")
    if len(text_content) > 50000:
        raise HTTPException(status_code=400, detail="Text content too long (max 50,000 characters)")
    
    return await _process_text_unified(text_content)

@app.post("/api/process-filepath")
async def process_filepath(data: Dict[str, str]):
    """
    Enhanced file path processing with consistent pipeline usage and streaming progress
    Uses identical processing pipeline as PDF upload and text input for consistency
    """
    file_path = data.get('file_path', '').strip()
    
    if not file_path:
        raise HTTPException(status_code=400, detail="File path is required")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Check if it's a PDF
    if not file_path.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Initialize performance tracker
    tracker = get_performance_tracker()
    session_id = tracker.create_session_id()
    
    async def generate_events() -> AsyncGenerator[str, None]:
        try:
            start_time = time.time()
            
            # Phase 1: Text Extraction & Semantic Analysis
            yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'extracting_text', 'development_mode': True, 'input_type': 'filepath'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Extract full text and get page count
            full_text = extract_text_from_pdf(file_path)
            
            if not full_text or len(full_text.strip()) < 10:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF or PDF is empty")
            
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            # Use SEMANTIC chunking (same as other endpoints)
            semantic_chunks = create_intelligent_chunks(full_text, total_pages)
            total_chunks = len(semantic_chunks)
            
            yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'semantic_chunking', 'enhancement': 'filepath_processing', 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            # Use real-time processing pipeline for immediate progress updates
            context_manager = LLMContextManager(max_context=5)
            async for event in execute_realtime_pipeline(
                semantic_chunks, full_text, total_pages, session_id,
                os.path.basename(file_path), os.path.getsize(file_path), start_time, "filepath",
                context_manager, process_chunk_with_context
            ):
                yield event
            yield f"data: {json.dumps({'event': 'merging_start', 'status': 'deduplicating', 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            # Collect all entries from chunks
            all_entries = []
            global_notes = None
            
            for chunk_result in chunk_results:
                if not chunk_result.get('error'):
                    entries = chunk_result.get('entries', [])
                    all_entries.extend(entries)
                    if not global_notes and chunk_result.get('global_notes'):
                        global_notes = chunk_result.get('global_notes')
            
            initial_count = len(all_entries)
            
            # Run full deduplication pipeline (same as other endpoints)
            yield f"data: {json.dumps({'event': 'dedup_progress', 'status': 'running_deduplication', 'initial_count': initial_count, 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            deduplicated_entries, dedup_stats = full_deduplication_pipeline(
                all_entries,
                fuzzy_threshold=Config.FUZZY_MATCH_THRESHOLD,
                enable_fuzzy=Config.ENABLE_FUZZY_DEDUP,
                enable_repair=Config.ENABLE_TRUNCATION_REPAIR
            )
            
            # Send deduplication stats
            yield f"data: {json.dumps({'event': 'dedup_complete', 'stats': dedup_stats, 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            # Phase 5: Enhanced Position Tracking (Fix Row Order)
            yield f"data: {json.dumps({'event': 'position_tracking_start', 'status': 'fixing_row_order', 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            # Apply Enhanced Method 1 position tracking (same as other endpoints)
            sorted_entries = fix_entry_ordering_enhanced_method_1(deduplicated_entries, full_text, semantic_chunks)
            
            # Add row numbers
            for i, entry in enumerate(sorted_entries, 1):
                entry['#'] = i
            
            yield f"data: {json.dumps({'event': 'position_tracking_complete', 'status': 'row_order_fixed', 'total_entries': len(sorted_entries), 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)

            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'development_mode': True, 'input_type': 'filepath'})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/process-unified")
async def process_unified(data: Dict[str, Any]):
    """
    Unified processing endpoint that handles all input types with consistent pipeline and output formatting
    Ensures identical processing across text input, file path, and PDF upload methods
    """
    input_type = data.get('input_type', '').lower()
    
    if input_type == 'text':
        text_content = data.get('content', '').strip()
        if not text_content:
            raise HTTPException(status_code=400, detail="Text content is required")
        if len(text_content) > 50000:
            raise HTTPException(status_code=400, detail="Text content too long (max 50,000 characters)")
        return await _process_text_unified(text_content)
    
    elif input_type == 'filepath':
        file_path = data.get('content', '').strip()
        if not file_path:
            raise HTTPException(status_code=400, detail="File path is required")
        return await _process_filepath_unified(file_path)
    
    else:
        raise HTTPException(status_code=400, detail="Invalid input_type. Must be 'text' or 'filepath'")


async def _process_pdf_upload_unified(content: bytes, filename: str):
    """Unified PDF upload processing with consistent pipeline"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(content)
        temp_pdf_path = temp_file.name
    
    try:
        # Initialize performance tracker
        tracker = get_performance_tracker()
        session_id = tracker.create_session_id()
        
        async def generate_events() -> AsyncGenerator[str, None]:
            try:
                start_time = time.time()
                
                # Phase 1: Analysis (identical to other endpoints)
                yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'extracting_text', 'input_type': 'pdf_upload'})}\\n\\n"
                await asyncio.sleep(0.1)
                
                full_text = extract_text_from_pdf(temp_pdf_path)
                reader = PdfReader(temp_pdf_path)
                total_pages = len(reader.pages)
                
                semantic_chunks = create_intelligent_chunks(full_text, total_pages)
                total_chunks = len(semantic_chunks)
                
                yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'semantic_chunking'})}\\n\\n"
                await asyncio.sleep(0.1)
                
                # Use real-time processing pipeline for immediate progress updates
                context_manager = LLMContextManager(max_context=5)
                async for event in execute_realtime_pipeline(
                    semantic_chunks, full_text, total_pages, session_id,
                    filename, len(content), start_time, "pdf_upload",
                    context_manager, process_chunk_with_context
                ):
                    yield event
                    
            except Exception as e:
                yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'input_type': 'pdf_upload'})}\\n\\n"
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
    except Exception as e:
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        raise


async def _process_text_unified(text_content: str):
    """Unified text processing with consistent pipeline"""
    # Initialize performance tracker
    tracker = get_performance_tracker()
    session_id = tracker.create_session_id()
    
    async def generate_events() -> AsyncGenerator[str, None]:
        try:
            start_time = time.time()
            
            # Phase 1: Analysis (identical to other endpoints)
            yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'analyzing_text', 'development_mode': True, 'input_type': 'text'})}\n\n"
            await asyncio.sleep(0.1)
            
            semantic_chunks = create_intelligent_chunks(text_content, total_pages=1)
            total_chunks = len(semantic_chunks)
            
            yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'semantic_chunking', 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            # Use real-time processing pipeline for immediate progress updates
            context_manager = LLMContextManager(max_context=5)
            async for event in execute_realtime_pipeline(
                semantic_chunks, text_content, 1, session_id, 
                "text_input.txt", len(text_content), start_time, "text",
                context_manager, process_chunk_with_context
            ):
                yield event
                
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'development_mode': True, 'input_type': 'text'})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def process_chunks_with_sse_progress(semantic_chunks, context_manager, batch_size, sse_yield_func):
    """
    Process chunks with real-time SSE progress reporting
    
    Args:
        semantic_chunks: Chunks to process
        context_manager: LLM context manager
        batch_size: Batch size for parallel processing
        sse_yield_func: Function to yield SSE events (async generator yield)
    
    Returns:
        Processed chunk results
    """
    # Clear any previous progress data
    if hasattr(process_chunks_batch_parallel, '_last_progress'):
        process_chunks_batch_parallel._last_progress = []
    
    # Process chunks in parallel batches with progress events
    chunk_results, progress_events = await process_chunks_batch_parallel(semantic_chunks, context_manager, batch_size, None, True)
    
    # Yield progress events via SSE
    for progress_event in progress_events:
        await sse_yield_func(f"data: {json.dumps(progress_event)}\n\n")
        await asyncio.sleep(0.05)  # Small delay for smooth updates
    
    return chunk_results


async def _process_filepath_unified(file_path: str):
    """Unified file path processing with consistent pipeline"""
    # Validation
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not file_path.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Initialize performance tracker
    tracker = get_performance_tracker()
    session_id = tracker.create_session_id()
    
    async def generate_events() -> AsyncGenerator[str, None]:
        try:
            start_time = time.time()
            
            # Phase 1: Analysis (identical to other endpoints)
            yield f"data: {json.dumps({'event': 'analysis_start', 'status': 'extracting_text', 'development_mode': True, 'input_type': 'filepath'})}\n\n"
            await asyncio.sleep(0.1)
            
            full_text = extract_text_from_pdf(file_path)
            if not full_text or len(full_text.strip()) < 10:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF or PDF is empty")
            
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            semantic_chunks = create_intelligent_chunks(full_text, total_pages)
            total_chunks = len(semantic_chunks)
            
            yield f"data: {json.dumps({'event': 'analysis_complete', 'total_chunks': total_chunks, 'strategy': 'semantic_chunking', 'development_mode': True})}\n\n"
            await asyncio.sleep(0.1)
            
            # Use real-time processing pipeline for immediate progress updates
            context_manager = LLMContextManager(max_context=5)
            async for event in execute_realtime_pipeline(
                semantic_chunks, full_text, total_pages, session_id,
                os.path.basename(file_path), os.path.getsize(file_path), start_time, "filepath",
                context_manager, process_chunk_with_context
            ):
                yield event
                
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'development_mode': True, 'input_type': 'filepath'})}\n\n"
    
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


@app.post("/api/test-context")
async def test_context_memory(data: Dict[str, Any]):
    """
    Test endpoint for context memory system
    
    Args:
        data: {"chunks": ["chunk1 content", "chunk2 content", ...]}
    """
    try:
        chunks = data.get('chunks', [])
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks provided")
        
        # Initialize context manager
        context_manager = LLMContextManager(max_context=3)
        
        results = []
        
        # Process chunks sequentially to test context building
        for i, chunk_content in enumerate(chunks):
            chunk = {
                'chunk_id': i + 1,
                'content': chunk_content
            }
            
            # Process with context
            result = await process_chunk_with_context(chunk, context_manager)
            results.append(result)
        
        return {
            "success": True,
            "results": results,
            "context_summary": context_manager.get_context_summary(),
            "message": f"Processed {len(chunks)} chunks with context memory"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context test failed: {str(e)}")


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

