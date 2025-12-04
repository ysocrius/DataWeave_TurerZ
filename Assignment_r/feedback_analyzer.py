"""
feedback_analyzer.py
--------------------
Analyzes user feedback comments to improve the auto-rating algorithm
"""

import re
from typing import Dict, List, Any, Optional
from learning_system import get_learning_system
from config import Config


class FeedbackAnalyzer:
    """Analyzes user feedback to improve auto-rating algorithm"""
    
    def __init__(self):
        self.learning_sys = get_learning_system()
        
        # Keyword mappings for different issues
        self.issue_keywords = {
            'speed': ['slow', 'fast', 'quick', 'time', 'speed', 'faster', 'slower', 'long'],
            'quality': ['wrong', 'accurate', 'quality', 'extraction', 'correct', 'incorrect', 'missed', 'missing'],
            'completeness': ['incomplete', 'partial', 'missing', 'missed', 'complete', 'all'],
            'deduplication': ['duplicate', 'repeated', 'same', 'copies'],
            'errors': ['error', 'failed', 'broken', 'issue', 'problem']
        }
        
        # Sentiment keywords
        self.positive_words = ['great', 'excellent', 'perfect', 'amazing', 'good', 'nice', 'love', 'awesome']
        self.negative_words = ['bad', 'poor', 'terrible', 'awful', 'hate', 'wrong', 'slow', 'missed']
    
    def analyze_comment(self, comment: str, user_rating: int, auto_rating: int, session_id: str) -> Dict[str, Any]:
        """Analyze a user comment and extract insights"""
        
        if not comment or not comment.strip():
            return {'has_content': False}
        
        comment_lower = comment.lower()
        
        # Extract keywords and issues
        keywords = self._extract_keywords(comment_lower)
        issues = self._classify_issues(keywords)
        sentiment = self._calculate_sentiment(comment_lower)
        
        # Analyze rating disagreement
        rating_diff = user_rating - auto_rating
        disagreement_type = self._analyze_disagreement(rating_diff, issues)
        
        analysis = {
            'has_content': True,
            'comment': comment,
            'keywords': keywords,
            'issues': issues,
            'sentiment': sentiment,
            'rating_diff': rating_diff,
            'disagreement_type': disagreement_type,
            'session_id': session_id,
            'user_rating': user_rating,
            'auto_rating': auto_rating
        }
        
        return analysis
    
    def _extract_keywords(self, comment: str) -> List[str]:
        """Extract relevant keywords from comment"""
        words = re.findall(r'\b\w+\b', comment.lower())
        
        # Find keywords that match our issue categories
        relevant_keywords = []
        for word in words:
            for category, keywords in self.issue_keywords.items():
                if word in keywords:
                    relevant_keywords.append(word)
        
        return list(set(relevant_keywords))  # Remove duplicates
    
    def _classify_issues(self, keywords: List[str]) -> List[str]:
        """Classify what issues the user is mentioning"""
        issues = []
        
        for category, category_keywords in self.issue_keywords.items():
            if any(keyword in category_keywords for keyword in keywords):
                issues.append(category)
        
        return issues
    
    def _calculate_sentiment(self, comment: str) -> float:
        """Calculate sentiment score from -1 (negative) to +1 (positive)"""
        positive_count = sum(1 for word in self.positive_words if word in comment)
        negative_count = sum(1 for word in self.negative_words if word in comment)
        
        total_words = len(comment.split())
        if total_words == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
    
    def _analyze_disagreement(self, rating_diff: int, issues: List[str]) -> str:
        """Analyze why user disagreed with auto-rating"""
        if rating_diff == 0:
            return 'agreement'
        elif rating_diff > 0:
            return f'user_rated_higher_{"+".join(issues) if issues else "unknown"}'
        else:
            return f'user_rated_lower_{"+".join(issues) if issues else "unknown"}'
    
    def analyze_feedback_batch(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Analyze recent user feedback for patterns"""
        
        if not self.learning_sys.is_connected():
            return []
        
        try:
            # Get recent user feedback (not auto-generated)
            feedback_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}user_feedback"]
            
            user_feedback = list(feedback_collection.find(
                {'feedback_id': {'$not': {'$regex': '^auto_'}}},
                {'session_id': 1, 'rating': 1, 'feedback_text': 1}
            ).sort('timestamp', -1).limit(limit))
            
            analyses = []
            
            for feedback in user_feedback:
                session_id = feedback.get('session_id')
                user_rating = feedback.get('rating', 0)
                comment = feedback.get('feedback_text', '')
                
                # Get corresponding auto-feedback
                auto_feedback = feedback_collection.find_one({
                    'session_id': session_id,
                    'feedback_id': {'$regex': '^auto_'}
                })
                
                auto_rating = auto_feedback.get('rating', 0) if auto_feedback else 0
                
                # Analyze this feedback
                analysis = self.analyze_comment(comment, user_rating, auto_rating, session_id)
                if analysis.get('has_content'):
                    analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            print(f"⚠️  Error analyzing feedback batch: {e}")
            return []
    
    def generate_improvement_suggestions(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate suggestions for improving the auto-rating algorithm"""
        
        if not analyses:
            return {'suggestions': [], 'patterns': []}
        
        # Count issues and patterns
        issue_counts = {}
        rating_diffs = []
        sentiment_scores = []
        
        for analysis in analyses:
            # Count issues mentioned
            for issue in analysis.get('issues', []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            rating_diffs.append(analysis.get('rating_diff', 0))
            sentiment_scores.append(analysis.get('sentiment', 0))
        
        # Generate suggestions
        suggestions = []
        patterns = []
        
        # Check if users consistently rate lower
        avg_rating_diff = sum(rating_diffs) / len(rating_diffs) if rating_diffs else 0
        if avg_rating_diff < -0.5:
            suggestions.append({
                'type': 'algorithm_too_optimistic',
                'description': f'Auto-rating is {abs(avg_rating_diff):.1f} stars too high on average',
                'action': 'Reduce overall rating by adjusting thresholds',
                'confidence': min(0.9, len(analyses) / 10)
            })
        
        # Check most common issues
        if issue_counts:
            most_common_issue = max(issue_counts.items(), key=lambda x: x[1])
            issue_name, count = most_common_issue
            
            if count >= 3:  # At least 3 mentions
                suggestions.append({
                    'type': f'{issue_name}_factor_adjustment',
                    'description': f'Users mention {issue_name} issues in {count} feedback',
                    'action': f'Increase weight of {issue_name} factor in rating calculation',
                    'confidence': min(0.8, count / len(analyses))
                })
        
        # Check sentiment vs rating difference
        negative_sentiment_count = sum(1 for s in sentiment_scores if s < -0.1)
        if negative_sentiment_count > len(analyses) * 0.6:  # >60% negative
            suggestions.append({
                'type': 'user_satisfaction_low',
                'description': f'{negative_sentiment_count}/{len(analyses)} feedback have negative sentiment',
                'action': 'Review and improve overall system performance',
                'confidence': 0.7
            })
        
        return {
            'total_analyzed': len(analyses),
            'avg_rating_diff': avg_rating_diff,
            'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            'issue_counts': issue_counts,
            'suggestions': suggestions,
            'patterns': patterns
        }
    
    def store_analysis_results(self, results: Dict[str, Any]) -> bool:
        """Store analysis results for tracking improvements"""
        
        if not self.learning_sys.is_connected():
            return False
        
        try:
            from datetime import datetime
            
            analysis_doc = {
                'timestamp': datetime.utcnow(),
                'type': 'feedback_analysis',
                'results': results
            }
            
            # Store in a new collection for feedback analysis
            analysis_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}feedback_analysis"]
            analysis_collection.insert_one(analysis_doc)
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error storing analysis results: {e}")
            return False


# Global instance
feedback_analyzer = FeedbackAnalyzer()


def analyze_user_feedback():
    """Convenience function to analyze recent user feedback"""
    analyses = feedback_analyzer.analyze_feedback_batch()
    results = feedback_analyzer.generate_improvement_suggestions(analyses)
    
    if results['suggestions']:
        print("🧠 Feedback Analysis Results:")
        print(f"   Analyzed {results['total_analyzed']} user feedback")
        print(f"   Avg rating difference: {results['avg_rating_diff']:+.1f}⭐")
        print(f"   Avg sentiment: {results['avg_sentiment']:+.2f}")
        
        print("\n💡 Improvement Suggestions:")
        for i, suggestion in enumerate(results['suggestions'], 1):
            print(f"   {i}. {suggestion['description']}")
            print(f"      Action: {suggestion['action']}")
            print(f"      Confidence: {suggestion['confidence']:.0%}")
        
        # Store results
        feedback_analyzer.store_analysis_results(results)
        
    return results


if __name__ == "__main__":
    analyze_user_feedback()