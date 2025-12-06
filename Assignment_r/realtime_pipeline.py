"""
Real-time Processing Pipeline for AI Document Processor
Provides true real-time SSE progress updates using asyncio.as_completed

This module extracts the real-time processing logic from the file path endpoint
to be reused by all input methods (PDF upload, text input, file path).
"""

import asyncio
import json
import time
import os
from typing import Dict, Any, List, AsyncGenerator, Optional

from config import Config
from utils_dedup import full_deduplication_pipeline
from position_tracker import fix_entry_ordering_enhanced_method_1
from performance_tracker import get_performance_tracker

async def _phase_context_init(batch_size: int) -> AsyncGenerator[str, None]:
    """
    Phase 2: Context Memory Initialization
    Sends SSE event to notify frontend that context manager is ready
    """
    yield f"data: {json.dumps({'event': 'context_initialized', 'max_context': 5, 'batch_size': batch_size, 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)

async def _phase_send_chunk_start_events(semantic_chunks: List[Dict[str, Any]], total_chunks: int) -> AsyncGenerator[str, None]:
    """
    Phase 3a: Send initial "started" events for ALL chunks upfront
    This shows all chunks in "Started" state in the UI immediately
    """
    for i, chunk in enumerate(semantic_chunks):
        chunk_num = i + 1
        yield f"data: {json.dumps({'event': 'chunk_start', 'chunk': chunk_num, 'total_chunks': total_chunks, 'page_range': chunk.get('page_range', '1'), 'section': chunk.get('header', f'Section {chunk_num}'), 'status': 'started', 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)

async def _phase_parallel_processing(
    semantic_chunks: List[Dict[str, Any]],
    total_chunks: int,
    context_manager,
    process_chunk_func
) -> AsyncGenerator[str, None]:
    """
    Phase 3b: Process chunks in parallel and send completion events IMMEDIATELY
    Uses asyncio.as_completed to emit events as soon as each chunk finishes
    Returns chunk_results via yielding a special data event
    """
    # Create async wrapper that tracks timing and chunk_id
    async def process_chunk_with_timing(chunk, chunk_id, semaphore):
        async with semaphore:
            chunk_start = time.time()
            result = await process_chunk_func(chunk, context_manager)
            result['processing_time'] = round(time.time() - chunk_start, 2)
            result['chunk_id'] = chunk_id
            return result
    
    # Process all chunks in parallel with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent LLM calls
    tasks = []
    for i, chunk in enumerate(semantic_chunks):
        task = asyncio.create_task(process_chunk_with_timing(chunk, i + 1, semaphore))
        tasks.append(task)
    
    # Collect results as they complete and emit SSE events IMMEDIATELY
    chunk_results = [None] * total_chunks  # Pre-allocate to maintain order
    
    for completed_task in asyncio.as_completed(tasks):
        chunk_result = await completed_task
        chunk_num = chunk_result.get('chunk_id', 0)
        chunk_results[chunk_num - 1] = chunk_result
        
        # Send completion event IMMEDIATELY when chunk finishes
        status = 'success' if not chunk_result.get('error') else 'error'
        yield f"data: {json.dumps({'event': 'chunk_complete', 'chunk': chunk_num, 'total_chunks': total_chunks, 'status': status, 'entries_count': chunk_result.get('entries_count', 0), 'processing_time': chunk_result.get('processing_time'), 'error': chunk_result.get('error'), 'development_mode': True})}\n\n"
        await asyncio.sleep(0.05)  # Small delay to ensure event is flushed
    
    # Yield chunk results as special internal event
    yield ('__CHUNK_RESULTS__', chunk_results)

async def _phase_deduplication(chunk_results: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """
    Phase 4: Deduplication
    Collects all entries and runs deduplication pipeline
    """
    yield f"data: {json.dumps({'event': 'merging_start', 'status': 'deduplicating', 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)
    
    # Collect all entries from chunks
    all_entries = []
    global_notes = None
    
    for chunk_result in chunk_results:
        if chunk_result and not chunk_result.get('error'):
            entries = chunk_result.get('entries', [])
            all_entries.extend(entries)
            if not global_notes and chunk_result.get('global_notes'):
                global_notes = chunk_result.get('global_notes')
    
    initial_count = len(all_entries)
    
    yield f"data: {json.dumps({'event': 'dedup_progress', 'status': 'running_deduplication', 'initial_count': initial_count, 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)
    
    deduplicated_entries, dedup_stats = full_deduplication_pipeline(
        all_entries,
        fuzzy_threshold=Config.FUZZY_MATCH_THRESHOLD,
        enable_fuzzy=Config.ENABLE_FUZZY_DEDUP,
        enable_repair=Config.ENABLE_TRUNCATION_REPAIR
    )
    
    yield f"data: {json.dumps({'event': 'dedup_complete', 'stats': dedup_stats, 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)
    
    # Yield dedup results as special internal event
    yield ('__DEDUP_RESULTS__', deduplicated_entries, global_notes, dedup_stats)

async def _phase_position_tracking(
    deduplicated_entries: List[Dict[str, Any]],
    full_text: str,
    semantic_chunks: List[Dict[str, Any]],
    global_notes: Optional[str]
) -> AsyncGenerator[str, None]:
    """
    Phase 5: Position Tracking
    Fixes row order and adds row numbers
    """
    yield f"data: {json.dumps({'event': 'position_tracking_start', 'status': 'fixing_row_order', 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)
    
    sorted_entries = fix_entry_ordering_enhanced_method_1(deduplicated_entries, full_text, semantic_chunks)
    
    # Add row numbers
    for i, entry in enumerate(sorted_entries, 1):
        entry['#'] = i
    
    yield f"data: {json.dumps({'event': 'position_tracking_complete', 'status': 'row_order_fixed', 'total_entries': len(sorted_entries), 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)
    
    # Add global notes to first entry
    if global_notes and len(sorted_entries) > 0:
        first_entry = sorted_entries[0]
        existing_comment = first_entry.get('Comments') or ''
        if existing_comment:
            first_entry['Comments'] = f"{global_notes} | {existing_comment}"
        else:
            first_entry['Comments'] = global_notes
    
    # Yield sorted entries as special internal event
    yield ('__SORTED_ENTRIES__', sorted_entries)

def _track_learning_system(
    session_id: str,
    filename: str,
    file_size: int,
    total_pages: int,
    chunk_results: List[Dict[str, Any]],
    dedup_stats: Dict[str, Any],
    sorted_entries: List[Dict[str, Any]],
    total_duration: float
):
    """
    Track performance in learning system
    """
    if Config.LEARNING_ENABLED:
        try:
            tracker = get_performance_tracker()
            processed_chunks = []
            for i, result in enumerate(chunk_results):
                if result:
                    processed_chunks.append({
                        "chunk_id": i + 1,
                        "entries_count": len(result.get('entries', [])),
                        "processing_time": result.get('processing_time', 0),
                        "status": "error" if result.get('error') else "success",
                        "error": result.get('error'),
                        "context_used": result.get('context_used', False)
                    })
            
            tracker.track_processing_session(
                session_id=session_id,
                filename=filename,
                file_size=file_size,
                total_pages=total_pages,
                chunk_size=0,
                chunk_overlap=0,
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

async def execute_realtime_pipeline(
    semantic_chunks: List[Dict[str, Any]],
    full_text: str,
    total_pages: int,
    session_id: str,
    filename: str,
    file_size: int,
    start_time: float,
    input_type: str,
    context_manager,
    process_chunk_func
) -> AsyncGenerator[str, None]:
    """
    Execute the processing pipeline with TRUE real-time SSE progress updates.
    
    Uses asyncio.as_completed to send chunk completion events immediately
    as each chunk finishes processing, rather than waiting for all chunks.
    
    Args:
        semantic_chunks: List of chunks to process
        full_text: Full extracted text (for position tracking)
        total_pages: Total number of pages
        session_id: Unique session ID for tracking
        filename: Name of the file being processed
        file_size: Size of the file in bytes
        start_time: Processing start time
        input_type: Type of input ('pdf_upload', 'text', 'filepath')
        context_manager: LLMContextManager instance
        process_chunk_func: Async function to process a single chunk
    
    Yields:
        SSE-formatted JSON events for real-time progress updates
    """
    total_chunks = len(semantic_chunks)
    batch_size = min(5, total_chunks)
    
    # Phase 2: Context Memory Initialization
    async for event in _phase_context_init(batch_size):
        yield event
    
    # Phase 3: Real-time Parallel Processing
    yield f"data: {json.dumps({'event': 'batch_processing_start', 'strategy': 'realtime_parallel', 'batch_size': batch_size, 'concurrency': min(3, total_chunks), 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)
    
    # Phase 3a: Send chunk start events
    async for event in _phase_send_chunk_start_events(semantic_chunks, total_chunks):
        yield event
    
    # Phase 3b: Process chunks and send completion events in real-time
    chunk_results = None
    async for event in _phase_parallel_processing(semantic_chunks, total_chunks, context_manager, process_chunk_func):
        if isinstance(event, tuple) and event[0] == '__CHUNK_RESULTS__':
            chunk_results = event[1]
        else:
            yield event
    
    # Send batch processing complete event
    successful_chunks = sum(1 for r in chunk_results if r and not r.get('error'))
    failed_chunks = len(chunk_results) - successful_chunks
    
    yield f"data: {json.dumps({'event': 'batch_processing_complete', 'successful': successful_chunks, 'failed': failed_chunks, 'context_summary': context_manager.get_context_summary(), 'development_mode': True})}\n\n"
    await asyncio.sleep(0.1)
    
    # Phase 4: Deduplication
    deduplicated_entries = None
    global_notes = None
    dedup_stats = None
    
    async for event in _phase_deduplication(chunk_results):
        if isinstance(event, tuple) and event[0] == '__DEDUP_RESULTS__':
            deduplicated_entries, global_notes, dedup_stats = event[1], event[2], event[3]
        else:
            yield event
    
    # Phase 5: Position Tracking
    sorted_entries = None
    
    async for event in _phase_position_tracking(deduplicated_entries, full_text, semantic_chunks, global_notes):
        if isinstance(event, tuple) and event[0] == '__SORTED_ENTRIES__':
            sorted_entries = event[1]
        else:
            yield event
    
    end_time = time.time()
    total_duration = round(end_time - start_time, 2)
    
    # Track performance in learning system
    _track_learning_system(
        session_id, filename, file_size, total_pages,
        chunk_results, dedup_stats, sorted_entries, total_duration
    )
    
    # Build chunks with processing results for frontend display
    chunks_with_results = []
    for i, chunk in enumerate(semantic_chunks):
        chunk_result = chunk_results[i] if i < len(chunk_results) and chunk_results[i] else {}
        chunks_with_results.append({
            "chunk_id": chunk.get('chunk_id', i + 1),
            "page_range": chunk.get('page_range', '1'),
            "header": chunk.get('header', f'Section {i + 1}'),
            "section_type": chunk.get('section_type', 'semantic_section'),
            "entries_count": chunk_result.get('entries_count', len(chunk_result.get('entries', []))),
            "processing_time": chunk_result.get('processing_time', 0),
            "status": 'success' if not chunk_result.get('error') else 'error',
            "error": chunk_result.get('error')
        })
    
    # Send final complete event
    final_result = {
        "event": "enhanced_complete",
        "session_id": session_id,
        "entries": sorted_entries,
        "global_notes": global_notes,
        "chunks": chunks_with_results,
        "chunk_results": chunk_results,
        "dedup_stats": dedup_stats,
        "context_summary": context_manager.get_context_summary(),
        "total_rows": len(sorted_entries),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "processing_time": total_duration,
        "processing_strategy": f"{input_type}_realtime_semantic_chunking",
        "batch_size": batch_size,
        "successful_chunks": successful_chunks,
        "failed_chunks": failed_chunks,
        "input_type": input_type,
        "development_mode": True
    }
    yield f"data: {json.dumps(final_result)}\n\n"