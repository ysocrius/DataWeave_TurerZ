"""
automated_feedback.py
---------------------
Automatically generates feedback ratings based on objective metrics
"""

from learning_models import UserFeedback
from learning_system import get_learning_system
import uuid


class AutomatedFeedbackGenerator:
    """Generates automated feedback for processing sessions"""
    
    def __init__(self):
        self.learning_sys = get_learning_system()
    
    def generate_feedback(self, session):
        """
        Generate automated feedback for a session
        
        Args:
            session: ProcessingSession object
            
        Returns:
            UserFeedback object
        """
        # Calculate rating
        rating = self._calculate_rating(session)
        
        # Generate feedback text
        feedback_text = self._generate_feedback_text(session, rating)
        
        # Create feedback object
        feedback = UserFeedback(
            feedback_id=f"auto_{uuid.uuid4().hex[:16]}",
            session_id=session.session_id,
            rating=int(round(rating)),
            feedback_text=feedback_text,
            corrections=[],
            corrections_count=0
        )
        
        return feedback
    
    def _calculate_rating(self, session):
        """Calculate 1-5 star rating"""
        
        # Factor 1: Quality Score (30%)
        quality_stars = 1 + (session.extraction_quality_score * 4)
        
        # Factor 2: Processing Speed (20%)
        speed_stars = self._calculate_speed_stars(session)
        
        # Factor 3: Data Completeness (25%)
        completeness_stars = self._calculate_completeness_stars(session)
        
        # Factor 4: Deduplication Efficiency (15%)
        dedup_stars = self._calculate_dedup_stars(session)
        
        # Factor 5: Error Rate (10%)
        error_stars = self._calculate_error_stars(session)
        
        # Weighted average
        final_rating = (
            quality_stars * 0.30 +
            speed_stars * 0.20 +
            completeness_stars * 0.25 +
            dedup_stars * 0.15 +
            error_stars * 0.10
        )
        
        # Round to nearest 0.5 and clamp
        final_rating = round(final_rating * 2) / 2
        final_rating = max(1.0, min(5.0, final_rating))
        
        return final_rating
    
    def _calculate_speed_stars(self, session):
        """Calculate speed rating"""
        expected_time = session.total_pages * 5
        actual_time = session.total_processing_time
        
        if expected_time == 0:
            return 4.0
        
        speed_ratio = expected_time / actual_time
        
        if speed_ratio >= 1.5:
            return 5.0
        elif speed_ratio >= 1.2:
            return 4.5
        elif speed_ratio >= 0.9:
            return 4.0
        elif speed_ratio >= 0.7:
            return 3.0
        else:
            return 2.0
    
    def _calculate_completeness_stars(self, session):
        """Calculate data completeness rating"""
        if session.total_pages == 0:
            return 4.0
        
        entries_per_page = session.final_entries_count / session.total_pages
        
        if 10 <= entries_per_page <= 30:
            return 5.0
        elif 5 <= entries_per_page < 10:
            return 4.0
        elif entries_per_page < 5:
            return 3.0
        elif entries_per_page > 50:
            return 3.5
        else:
            return 4.5
    
    def _calculate_dedup_stars(self, session):
        """Calculate deduplication efficiency rating"""
        initial = session.dedup_stats.initial_count
        final = session.dedup_stats.final_count
        
        if initial == 0:
            return 4.0
        
        dedup_rate = (initial - final) / initial
        
        if 0.20 <= dedup_rate <= 0.40:
            return 5.0
        elif 0.10 <= dedup_rate < 0.20:
            return 4.0
        elif dedup_rate < 0.10:
            return 3.5
        elif 0.40 < dedup_rate <= 0.60:
            return 4.0
        else:
            return 3.0
    
    def _calculate_error_stars(self, session):
        """Calculate error rate rating"""
        if session.total_chunks == 0:
            return 5.0
        
        error_count = sum(1 for chunk in session.chunk_results if chunk.status == 'error')
        error_rate = error_count / session.total_chunks
        
        if error_rate == 0:
            return 5.0
        elif error_rate < 0.1:
            return 4.5
        elif error_rate < 0.2:
            return 4.0
        elif error_rate < 0.3:
            return 3.0
        else:
            return 2.0
    
    def _generate_feedback_text(self, session, rating):
        """Generate descriptive feedback text"""
        feedback_parts = []
        
        # Quality assessment
        if session.extraction_quality_score >= 0.9:
            feedback_parts.append("excellent extraction quality")
        elif session.extraction_quality_score >= 0.8:
            feedback_parts.append("good extraction quality")
        elif session.extraction_quality_score >= 0.7:
            feedback_parts.append("acceptable extraction quality")
        else:
            feedback_parts.append("extraction quality needs improvement")
        
        # Speed assessment
        expected_time = session.total_pages * 5
        if session.total_processing_time < expected_time * 0.8:
            feedback_parts.append("fast processing")
        elif session.total_processing_time > expected_time * 1.3:
            feedback_parts.append("slower than expected")
        
        # Deduplication
        if session.dedup_stats.initial_count > 0:
            dedup_rate = ((session.dedup_stats.initial_count - session.dedup_stats.final_count) / 
                         session.dedup_stats.initial_count)
            if 0.20 <= dedup_rate <= 0.40:
                feedback_parts.append("optimal deduplication")
            elif dedup_rate > 0.50:
                feedback_parts.append("high deduplication rate")
        
        # Completeness
        entries_per_page = session.final_entries_count / session.total_pages if session.total_pages > 0 else 0
        if entries_per_page >= 15:
            feedback_parts.append(f"comprehensive data extraction ({session.final_entries_count} entries)")
        
        # Combine
        if feedback_parts:
            feedback = ", ".join(feedback_parts) + "."
            feedback = feedback[0].upper() + feedback[1:]
        else:
            feedback = "Processing completed successfully."
        
        return feedback
    
    def submit_automated_feedback(self, session):
        """Generate and submit automated feedback"""
        if not self.learning_sys.is_connected():
            return False
        
        feedback = self.generate_feedback(session)
        success = self.learning_sys.store_user_feedback(feedback)
        
        if success:
            print(f"✅ Auto-feedback: {feedback.rating}⭐ - {feedback.feedback_text}")
        
        return success


# Global instance
feedback_generator = AutomatedFeedbackGenerator()


def generate_and_submit_feedback(session):
    """Convenience function to generate and submit feedback"""
    return feedback_generator.submit_automated_feedback(session)
