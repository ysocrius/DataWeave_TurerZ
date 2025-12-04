"""
pattern_recognizer.py
---------------------
Recognizes patterns in user feedback to identify systematic issues
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
from learning_system import get_learning_system
from config import Config


class PatternRecognizer:
    """Recognizes patterns in user feedback vs session metrics"""
    
    def __init__(self):
        self.learning_sys = get_learning_system()
        self.min_sample_size = 3
        self.min_confidence = 0.6
    
    def find_feedback_patterns(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Find patterns in user feedback over the specified time period"""
        
        if not self.learning_sys.is_connected():
            return []
        
        try:
            # Get feedback with session data
            feedback_sessions = self._get_feedback_with_sessions(days_back)
            
            if len(feedback_sessions) < self.min_sample_size:
                return []
            
            patterns = []
            
            # Speed-related patterns
            speed_patterns = self._analyze_speed_patterns(feedback_sessions)
            patterns.extend(speed_patterns)
            
            # Quality-related patterns
            quality_patterns = self._analyze_quality_patterns(feedback_sessions)
            patterns.extend(quality_patterns)
            
            # Document type patterns
            doc_type_patterns = self._analyze_document_type_patterns(feedback_sessions)
            patterns.extend(doc_type_patterns)
            
            # Rating disagreement patterns
            disagreement_patterns = self._analyze_disagreement_patterns(feedback_sessions)
            patterns.extend(disagreement_patterns)
            
            # Filter by confidence
            high_confidence_patterns = [p for p in patterns if p['confidence'] >= self.min_confidence]
            
            return high_confidence_patterns
            
        except Exception as e:
            print(f"⚠️  Error finding feedback patterns: {e}")
            return []
    
    def _get_feedback_with_sessions(self, days_back: int) -> List[Dict[str, Any]]:
        """Get user feedback with corresponding session data"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Get user feedback (not auto-generated)
        feedback_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}user_feedback"]
        user_feedback = list(feedback_collection.find({
            'feedback_id': {'$not': {'$regex': '^auto_'}},
            'timestamp': {'$gte': cutoff_date}
        }))
        
        # Get corresponding sessions
        sessions_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}sessions"]
        
        feedback_sessions = []
        
        for feedback in user_feedback:
            session_id = feedback.get('session_id')
            if session_id:
                session = sessions_collection.find_one({'session_id': session_id})
                if session:
                    # Get auto-feedback for comparison
                    auto_feedback = feedback_collection.find_one({
                        'session_id': session_id,
                        'feedback_id': {'$regex': '^auto_'}
                    })
                    
                    combined = {
                        'user_feedback': feedback,
                        'session': session,
                        'auto_feedback': auto_feedback
                    }
                    feedback_sessions.append(combined)
        
        return feedback_sessions
    
    def _analyze_speed_patterns(self, feedback_sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns related to processing speed"""
        
        patterns = []
        
        # Group by speed complaints
        speed_complaints = []
        speed_praises = []
        
        for fs in feedback_sessions:
            comment = fs['user_feedback'].get('feedback_text', '').lower()
            processing_time = fs['session'].get('total_processing_time', 0)
            user_rating = fs['user_feedback'].get('rating', 0)
            auto_rating = fs['auto_feedback'].get('rating', 0) if fs['auto_feedback'] else 0
            
            # Check for speed-related keywords
            speed_negative = any(word in comment for word in ['slow', 'long', 'time', 'wait'])
            speed_positive = any(word in comment for word in ['fast', 'quick', 'rapid'])
            
            if speed_negative and user_rating < auto_rating:
                speed_complaints.append({
                    'processing_time': processing_time,
                    'rating_diff': user_rating - auto_rating,
                    'comment': comment
                })
            elif speed_positive and user_rating >= auto_rating:
                speed_praises.append({
                    'processing_time': processing_time,
                    'rating_diff': user_rating - auto_rating,
                    'comment': comment
                })
        
        # Pattern: Users complain about speed when processing time > threshold
        if len(speed_complaints) >= self.min_sample_size:
            times = [sc['processing_time'] for sc in speed_complaints]
            avg_complaint_time = statistics.mean(times)
            
            # Find threshold where complaints start
            all_times = [fs['session'].get('total_processing_time', 0) for fs in feedback_sessions]
            avg_all_times = statistics.mean(all_times) if all_times else 0
            
            if avg_complaint_time > avg_all_times * 1.5:  # 50% slower than average
                patterns.append({
                    'type': 'speed_threshold_pattern',
                    'description': f'Users complain about speed when processing time > {avg_complaint_time:.1f}s',
                    'confidence': min(0.9, len(speed_complaints) / len(feedback_sessions)),
                    'sample_size': len(speed_complaints),
                    'threshold': avg_complaint_time,
                    'recommendation': 'Increase speed weight in rating algorithm',
                    'data': {
                        'avg_complaint_time': avg_complaint_time,
                        'avg_overall_time': avg_all_times,
                        'complaint_count': len(speed_complaints)
                    }
                })
        
        return patterns
    
    def _analyze_quality_patterns(self, feedback_sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns related to extraction quality"""
        
        patterns = []
        
        # Group by quality complaints
        quality_complaints = []
        quality_praises = []
        
        for fs in feedback_sessions:
            comment = fs['user_feedback'].get('feedback_text', '').lower()
            quality_score = fs['session'].get('extraction_quality_score', 0)
            user_rating = fs['user_feedback'].get('rating', 0)
            auto_rating = fs['auto_feedback'].get('rating', 0) if fs['auto_feedback'] else 0
            
            # Check for quality-related keywords
            quality_negative = any(word in comment for word in ['wrong', 'missed', 'incorrect', 'bad', 'poor'])
            quality_positive = any(word in comment for word in ['accurate', 'correct', 'perfect', 'excellent'])
            
            if quality_negative and user_rating < auto_rating:
                quality_complaints.append({
                    'quality_score': quality_score,
                    'rating_diff': user_rating - auto_rating,
                    'comment': comment
                })
            elif quality_positive and user_rating >= auto_rating:
                quality_praises.append({
                    'quality_score': quality_score,
                    'rating_diff': user_rating - auto_rating,
                    'comment': comment
                })
        
        # Pattern: Users complain about quality when score < threshold
        if len(quality_complaints) >= self.min_sample_size:
            scores = [qc['quality_score'] for qc in quality_complaints]
            avg_complaint_score = statistics.mean(scores)
            
            # Find threshold where complaints start
            all_scores = [fs['session'].get('extraction_quality_score', 0) for fs in feedback_sessions]
            avg_all_scores = statistics.mean(all_scores) if all_scores else 0
            
            if avg_complaint_score < avg_all_scores * 0.9:  # 10% below average
                patterns.append({
                    'type': 'quality_threshold_pattern',
                    'description': f'Users complain about quality when score < {avg_complaint_score:.2f}',
                    'confidence': min(0.9, len(quality_complaints) / len(feedback_sessions)),
                    'sample_size': len(quality_complaints),
                    'threshold': avg_complaint_score,
                    'recommendation': 'Increase quality threshold for higher ratings',
                    'data': {
                        'avg_complaint_score': avg_complaint_score,
                        'avg_overall_score': avg_all_scores,
                        'complaint_count': len(quality_complaints)
                    }
                })
        
        return patterns
    
    def _analyze_document_type_patterns(self, feedback_sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns based on document characteristics"""
        
        patterns = []
        
        # Group by document size
        small_docs = []  # < 100KB
        medium_docs = []  # 100KB - 500KB
        large_docs = []  # > 500KB
        
        for fs in feedback_sessions:
            file_size = fs['session'].get('file_size', 0)
            user_rating = fs['user_feedback'].get('rating', 0)
            auto_rating = fs['auto_feedback'].get('rating', 0) if fs['auto_feedback'] else 0
            rating_diff = user_rating - auto_rating
            
            if file_size < 100000:
                small_docs.append(rating_diff)
            elif file_size < 500000:
                medium_docs.append(rating_diff)
            else:
                large_docs.append(rating_diff)
        
        # Check for size-based rating patterns
        doc_groups = [
            ('small', small_docs, '< 100KB'),
            ('medium', medium_docs, '100-500KB'),
            ('large', large_docs, '> 500KB')
        ]
        
        for group_name, rating_diffs, size_desc in doc_groups:
            if len(rating_diffs) >= self.min_sample_size:
                avg_diff = statistics.mean(rating_diffs)
                
                if abs(avg_diff) > 0.5:  # Significant difference
                    direction = 'higher' if avg_diff > 0 else 'lower'
                    patterns.append({
                        'type': f'document_size_pattern_{group_name}',
                        'description': f'Users rate {direction} for {size_desc} documents (avg diff: {avg_diff:+.1f})',
                        'confidence': min(0.8, len(rating_diffs) / 10),
                        'sample_size': len(rating_diffs),
                        'avg_rating_diff': avg_diff,
                        'document_size_category': group_name,
                        'recommendation': f'Adjust rating algorithm for {size_desc} documents',
                        'data': {
                            'size_category': group_name,
                            'size_description': size_desc,
                            'avg_rating_diff': avg_diff,
                            'sample_count': len(rating_diffs)
                        }
                    })
        
        return patterns
    
    def _analyze_disagreement_patterns(self, feedback_sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in rating disagreements"""
        
        patterns = []
        
        # Categorize disagreements
        strong_disagreements = []  # |diff| >= 2
        mild_disagreements = []    # 1 <= |diff| < 2
        agreements = []            # |diff| < 1
        
        for fs in feedback_sessions:
            user_rating = fs['user_feedback'].get('rating', 0)
            auto_rating = fs['auto_feedback'].get('rating', 0) if fs['auto_feedback'] else 0
            
            if auto_rating > 0:  # Only if we have auto rating
                diff = abs(user_rating - auto_rating)
                
                if diff >= 2:
                    strong_disagreements.append(fs)
                elif diff >= 1:
                    mild_disagreements.append(fs)
                else:
                    agreements.append(fs)
        
        total_with_auto = len(strong_disagreements) + len(mild_disagreements) + len(agreements)
        
        if total_with_auto >= self.min_sample_size:
            disagreement_rate = (len(strong_disagreements) + len(mild_disagreements)) / total_with_auto
            
            if disagreement_rate > 0.3:  # >30% disagreement
                patterns.append({
                    'type': 'high_disagreement_pattern',
                    'description': f'High disagreement rate: {disagreement_rate:.1%} of users disagree with auto-ratings',
                    'confidence': min(0.9, total_with_auto / 20),
                    'sample_size': total_with_auto,
                    'disagreement_rate': disagreement_rate,
                    'recommendation': 'Review and recalibrate auto-rating algorithm',
                    'data': {
                        'strong_disagreements': len(strong_disagreements),
                        'mild_disagreements': len(mild_disagreements),
                        'agreements': len(agreements),
                        'total_comparisons': total_with_auto
                    }
                })
        
        return patterns
    
    def get_pattern_summary(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of all patterns"""
        
        if not patterns:
            return {'total_patterns': 0, 'categories': {}}
        
        categories = defaultdict(list)
        
        for pattern in patterns:
            category = pattern['type'].split('_')[0]  # First word of type
            categories[category].append(pattern)
        
        summary = {
            'total_patterns': len(patterns),
            'categories': dict(categories),
            'high_confidence_count': len([p for p in patterns if p['confidence'] > 0.8]),
            'avg_confidence': statistics.mean([p['confidence'] for p in patterns]),
            'total_sample_size': sum([p['sample_size'] for p in patterns]),
            'recommendations': [p['recommendation'] for p in patterns if 'recommendation' in p]
        }
        
        return summary


# Global instance
pattern_recognizer = PatternRecognizer()


def analyze_feedback_patterns():
    """Main function to analyze feedback patterns"""
    
    print("\n" + "="*60)
    print("Feedback Pattern Recognition")
    print("="*60)
    
    patterns = pattern_recognizer.find_feedback_patterns()
    
    if not patterns:
        print("⚠️  No significant patterns found in user feedback")
        return []
    
    print(f"✅ Found {len(patterns)} feedback patterns:")
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n{i}. {pattern['type'].replace('_', ' ').title()}")
        print(f"   Description: {pattern['description']}")
        print(f"   Confidence: {pattern['confidence']:.0%}")
        print(f"   Sample Size: {pattern['sample_size']}")
        if 'recommendation' in pattern:
            print(f"   Recommendation: {pattern['recommendation']}")
    
    # Generate summary
    summary = pattern_recognizer.get_pattern_summary(patterns)
    
    print(f"\n📊 Pattern Summary:")
    print(f"   Total Patterns: {summary['total_patterns']}")
    print(f"   High Confidence: {summary['high_confidence_count']}")
    print(f"   Average Confidence: {summary['avg_confidence']:.0%}")
    print(f"   Total Sample Size: {summary['total_sample_size']}")
    
    return patterns


if __name__ == "__main__":
    analyze_feedback_patterns()