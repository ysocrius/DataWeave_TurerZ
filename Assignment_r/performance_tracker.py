"""
performance_tracker.py
----------------------
Performance tracking and quality metrics for the learning system
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from learning_models import (
    ProcessingSession, PerformanceMetric, ChunkResult,
    DeduplicationStats, ProcessingStatus, DocumentType
)
from learning_system import get_learning_system
from config import Config


class PerformanceTracker:
    """Track and analyze processing performance"""
    
    def __init__(self):
        self.learning_system = get_learning_system()
    
    def create_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{uuid.uuid4().hex[:16]}"
    
    def track_processing_session(
        self,
        session_id: str,
        filename: str,
        file_size: int,
        total_pages: int,
        chunk_size: int,
        chunk_overlap: int,
        llm_model: str,
        temperature: float,
        fuzzy_threshold: float,
        chunk_results: List[Dict[str, Any]],
        dedup_stats: Dict[str, Any],
        final_entries_count: int,
        total_processing_time: float,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> bool:
        """Track a complete processing session"""
        
        if not self.learning_system.is_connected():
            return False
        
        try:
            # Convert chunk results to ChunkResult models
            chunk_result_models = [
                ChunkResult(
                    chunk_id=cr.get('chunk_id', 0),
                    char_range=cr.get('char_range'),
                    page_range=cr.get('page_range'),
                    entries_count=cr.get('entries_count', 0),
                    processing_time=cr.get('processing_time', 0.0),
                    content_length=cr.get('content_length', 0),
                    status=cr.get('status', 'unknown'),
                    error=cr.get('error')
                )
                for cr in chunk_results
            ]
            
            # Convert dedup stats to DeduplicationStats model
            dedup_stats_model = DeduplicationStats(
                initial_count=dedup_stats.get('initial_count', 0),
                after_exact_dedup=dedup_stats.get('after_exact_dedup', 0),
                after_fuzzy_dedup=dedup_stats.get('after_fuzzy_dedup', 0),
                after_consolidation=dedup_stats.get('after_consolidation', 0),
                final_count=dedup_stats.get('final_count', 0),
                exact_duplicates_removed=dedup_stats.get('exact_duplicates_removed', 0),
                fuzzy_duplicates_removed=dedup_stats.get('fuzzy_duplicates_removed', 0),
                entries_consolidated=dedup_stats.get('entries_consolidated', 0),
                truncation_repairs=dedup_stats.get('truncation_repairs', 0)
            )
            
            # Calculate average chunk time
            total_chunk_time = sum(cr.processing_time for cr in chunk_result_models)
            avg_chunk_time = total_chunk_time / len(chunk_result_models) if chunk_result_models else 0
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                chunk_results=chunk_result_models,
                dedup_stats=dedup_stats_model,
                final_entries_count=final_entries_count
            )
            
            # Create processing session
            session = ProcessingSession(
                session_id=session_id,
                filename=filename,
                file_size=file_size,
                total_pages=total_pages,
                document_type=DocumentType.UNKNOWN,  # Will be classified separately
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                llm_model=llm_model,
                temperature=temperature,
                fuzzy_threshold=fuzzy_threshold,
                total_chunks=len(chunk_result_models),
                chunk_results=chunk_result_models,
                dedup_stats=dedup_stats_model,
                final_entries_count=final_entries_count,
                total_processing_time=total_processing_time,
                avg_chunk_time=avg_chunk_time,
                extraction_quality_score=quality_score,
                status=ProcessingStatus(status),
                error_message=error_message
            )
            
            # Store session
            success = self.learning_system.store_processing_session(session)
            
            if success:
                # Also store as performance metric
                self._store_performance_metric(session)
                
                # Generate and submit automated feedback if enabled
                if Config.AUTOMATED_FEEDBACK_ENABLED:
                    try:
                        from automated_feedback import generate_and_submit_feedback
                        generate_and_submit_feedback(session)
                    except Exception as e:
                        print(f"⚠️  Automated feedback failed: {e}")
            
            return success
            
        except Exception as e:
            print(f"⚠️  Error tracking processing session: {e}")
            return False
    
    def _calculate_quality_score(
        self,
        chunk_results: List[ChunkResult],
        dedup_stats: DeduplicationStats,
        final_entries_count: int
    ) -> float:
        """
        Calculate quality score based on multiple factors
        Score range: 0.0 - 1.0
        """
        try:
            # Factor 1: Chunk success rate (40%)
            successful_chunks = sum(1 for cr in chunk_results if cr.status == 'success')
            chunk_success_rate = successful_chunks / len(chunk_results) if chunk_results else 0
            chunk_score = chunk_success_rate * 0.4
            
            # Factor 2: Deduplication efficiency (30%)
            # Good deduplication should remove 20-40% of entries
            if dedup_stats.initial_count > 0:
                dedup_rate = (dedup_stats.initial_count - dedup_stats.final_count) / dedup_stats.initial_count
                # Optimal range: 0.2-0.4 (20-40% reduction)
                if 0.2 <= dedup_rate <= 0.4:
                    dedup_score = 1.0
                elif dedup_rate < 0.2:
                    dedup_score = dedup_rate / 0.2  # Scale up to 1.0
                else:
                    dedup_score = max(0, 1.0 - (dedup_rate - 0.4) / 0.3)  # Scale down from 1.0
            else:
                dedup_score = 0.5  # Neutral score if no data
            dedup_score *= 0.3
            
            # Factor 3: Data extraction completeness (30%)
            # More entries generally means better extraction (up to a point)
            if final_entries_count > 0:
                # Assume 50-150 entries is optimal range
                if 50 <= final_entries_count <= 150:
                    extraction_score = 1.0
                elif final_entries_count < 50:
                    extraction_score = final_entries_count / 50
                else:
                    extraction_score = max(0.5, 1.0 - (final_entries_count - 150) / 300)
            else:
                extraction_score = 0
            extraction_score *= 0.3
            
            # Total quality score
            quality_score = chunk_score + dedup_score + extraction_score
            
            return round(quality_score, 3)
            
        except Exception as e:
            print(f"⚠️  Error calculating quality score: {e}")
            return 0.5  # Return neutral score on error
    
    def _store_performance_metric(self, session: ProcessingSession) -> bool:
        """Store performance metric from session"""
        try:
            # Calculate deduplication efficiency
            if session.dedup_stats.initial_count > 0:
                dedup_efficiency = (
                    (session.dedup_stats.initial_count - session.dedup_stats.final_count) /
                    session.dedup_stats.initial_count * 100
                )
            else:
                dedup_efficiency = 0
            
            # Count errors
            error_count = sum(1 for cr in session.chunk_results if cr.status == 'error')
            
            metric = PerformanceMetric(
                metric_id=f"metric_{uuid.uuid4().hex[:16]}",
                session_id=session.session_id,
                processing_time=session.total_processing_time,
                entries_extracted=session.final_entries_count,
                quality_score=session.extraction_quality_score,
                user_satisfaction=session.user_rating,
                chunk_count=session.total_chunks,
                deduplication_efficiency=round(dedup_efficiency, 2),
                error_count=error_count
            )
            
            return self.learning_system.store_performance_metric(metric)
            
        except Exception as e:
            print(f"⚠️  Error storing performance metric: {e}")
            return False
    
    def get_performance_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get performance summary for a session"""
        if not self.learning_system.is_connected():
            return None
        
        try:
            session = self.learning_system.get_processing_session(session_id)
            if not session:
                return None
            
            return {
                'session_id': session.session_id,
                'filename': session.filename,
                'total_processing_time': session.total_processing_time,
                'avg_chunk_time': session.avg_chunk_time,
                'quality_score': session.extraction_quality_score,
                'final_entries': session.final_entries_count,
                'dedup_efficiency': round(
                    (session.dedup_stats.initial_count - session.dedup_stats.final_count) /
                    session.dedup_stats.initial_count * 100, 2
                ) if session.dedup_stats.initial_count > 0 else 0,
                'status': session.status,
                'user_rating': session.user_rating
            }
            
        except Exception as e:
            print(f"⚠️  Error getting performance summary: {e}")
            return None


# Global performance tracker instance
performance_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    return performance_tracker
