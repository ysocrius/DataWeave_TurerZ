"""
algorithm_tuner.py
------------------
Automatically tunes the auto-rating algorithm based on user feedback patterns
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from learning_system import get_learning_system
from config import Config


class AlgorithmTuner:
    """Tunes algorithm parameters based on feedback learning"""
    
    def __init__(self):
        self.learning_sys = get_learning_system()
        
        # Current algorithm weights (from automated_feedback.py)
        self.current_weights = {
            "quality": 0.30,
            "speed": 0.20,
            "completeness": 0.25,
            "deduplication": 0.15,
            "error_rate": 0.10
        }
        
        # Current rating thresholds
        self.current_thresholds = {
            5: {"quality": 0.90, "speed": 0.85, "completeness": 0.90},
            4: {"quality": 0.80, "speed": 0.70, "completeness": 0.80},
            3: {"quality": 0.70, "speed": 0.60, "completeness": 0.70},
            2: {"quality": 0.60, "speed": 0.50, "completeness": 0.60},
            1: {"quality": 0.50, "speed": 0.40, "completeness": 0.50}
        }
        
        # Minimum confidence required for changes
        self.min_confidence = 0.6
        self.min_sample_size = 5
    
    def analyze_feedback_patterns(self) -> List[Dict[str, Any]]:
        """Analyze feedback patterns to identify algorithm improvements"""
        
        if not self.learning_sys.is_connected():
            return []
        
        try:
            # Get feedback analysis results
            analysis_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}feedback_analysis"]
            recent_analyses = list(analysis_collection.find().sort('timestamp', -1).limit(10))
            
            patterns = []
            
            if recent_analyses:
                # Aggregate data from recent analyses
                total_analyzed = 0
                total_rating_diff = 0
                total_sentiment = 0
                issue_counts = {}
                
                for analysis in recent_analyses:
                    results = analysis.get('results', {})
                    total_analyzed += results.get('total_analyzed', 0)
                    total_rating_diff += results.get('avg_rating_diff', 0) * results.get('total_analyzed', 0)
                    total_sentiment += results.get('avg_sentiment', 0) * results.get('total_analyzed', 0)
                    
                    for issue, count in results.get('issue_counts', {}).items():
                        issue_counts[issue] = issue_counts.get(issue, 0) + count
                
                if total_analyzed > 0:
                    avg_rating_diff = total_rating_diff / total_analyzed
                    avg_sentiment = total_sentiment / total_analyzed
                    
                    # Pattern 1: System too optimistic (users rate lower)
                    if avg_rating_diff < -0.5 and total_analyzed >= self.min_sample_size:
                        patterns.append({
                            'type': 'system_too_optimistic',
                            'description': f'Auto-ratings are {abs(avg_rating_diff):.1f} stars too high on average',
                            'confidence': min(0.9, total_analyzed / 20),
                            'adjustment': {
                                'action': 'lower_thresholds',
                                'amount': min(0.05, abs(avg_rating_diff) * 0.02)
                            },
                            'sample_size': total_analyzed
                        })
                    
                    # Pattern 2: System too pessimistic (users rate higher)
                    elif avg_rating_diff > 0.5 and total_analyzed >= self.min_sample_size:
                        patterns.append({
                            'type': 'system_too_pessimistic',
                            'description': f'Auto-ratings are {avg_rating_diff:.1f} stars too low on average',
                            'confidence': min(0.9, total_analyzed / 20),
                            'adjustment': {
                                'action': 'raise_thresholds',
                                'amount': min(0.05, avg_rating_diff * 0.02)
                            },
                            'sample_size': total_analyzed
                        })
                    
                    # Pattern 3: Specific issue focus
                    if issue_counts:
                        most_common_issue = max(issue_counts.items(), key=lambda x: x[1])
                        issue_name, count = most_common_issue
                        
                        if count >= 3 and count / total_analyzed > 0.3:  # >30% mention this issue
                            patterns.append({
                                'type': f'{issue_name}_weight_adjustment',
                                'description': f'Users frequently mention {issue_name} issues ({count}/{total_analyzed} feedback)',
                                'confidence': min(0.8, count / total_analyzed),
                                'adjustment': {
                                    'action': 'increase_weight',
                                    'factor': issue_name,
                                    'amount': 0.05
                                },
                                'sample_size': total_analyzed
                            })
                    
                    # Pattern 4: Sentiment-based adjustment
                    if avg_sentiment < -0.2 and total_analyzed >= self.min_sample_size:
                        patterns.append({
                            'type': 'negative_sentiment_trend',
                            'description': f'Overall negative sentiment ({avg_sentiment:.2f}) suggests system overconfidence',
                            'confidence': 0.7,
                            'adjustment': {
                                'action': 'conservative_tuning',
                                'amount': 0.03
                            },
                            'sample_size': total_analyzed
                        })
            
            return patterns
            
        except Exception as e:
            print(f"⚠️  Error analyzing feedback patterns: {e}")
            return []
    
    def apply_algorithm_improvements(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply algorithm improvements based on patterns"""
        
        if not patterns:
            return {'applied': 0, 'changes': []}
        
        changes = []
        applied_count = 0
        
        for pattern in patterns:
            if pattern['confidence'] >= self.min_confidence:
                change = self._apply_single_pattern(pattern)
                if change:
                    changes.append(change)
                    applied_count += 1
        
        # Store tuning history
        if changes:
            self._store_tuning_history(changes)
        
        return {
            'applied': applied_count,
            'changes': changes,
            'new_weights': self.current_weights.copy(),
            'new_thresholds': self.current_thresholds.copy()
        }
    
    def _apply_single_pattern(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply a single pattern adjustment"""
        
        adjustment = pattern.get('adjustment', {})
        action = adjustment.get('action')
        
        if action == 'lower_thresholds':
            amount = adjustment.get('amount', 0.02)
            return self._adjust_thresholds(-amount, pattern)
        
        elif action == 'raise_thresholds':
            amount = adjustment.get('amount', 0.02)
            return self._adjust_thresholds(amount, pattern)
        
        elif action == 'increase_weight':
            factor = adjustment.get('factor')
            amount = adjustment.get('amount', 0.05)
            return self._adjust_weight(factor, amount, pattern)
        
        elif action == 'conservative_tuning':
            amount = adjustment.get('amount', 0.03)
            return self._apply_conservative_tuning(amount, pattern)
        
        return None
    
    def _adjust_thresholds(self, amount: float, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust rating thresholds"""
        
        old_thresholds = self.current_thresholds.copy()
        
        for rating in self.current_thresholds:
            for metric in self.current_thresholds[rating]:
                new_value = self.current_thresholds[rating][metric] + amount
                # Keep within reasonable bounds
                self.current_thresholds[rating][metric] = max(0.3, min(0.95, new_value))
        
        return {
            'type': 'threshold_adjustment',
            'pattern': pattern['type'],
            'amount': amount,
            'old_thresholds': old_thresholds,
            'new_thresholds': self.current_thresholds.copy(),
            'confidence': pattern['confidence']
        }
    
    def _adjust_weight(self, factor: str, amount: float, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust weight for a specific factor"""
        
        if factor not in self.current_weights:
            return None
        
        old_weights = self.current_weights.copy()
        
        # Increase target factor weight
        self.current_weights[factor] = min(0.5, self.current_weights[factor] + amount)
        
        # Redistribute other weights proportionally
        remaining_weight = 1.0 - self.current_weights[factor]
        other_factors = [f for f in self.current_weights if f != factor]
        total_other_weight = sum(self.current_weights[f] for f in other_factors)
        
        if total_other_weight > 0:
            for f in other_factors:
                self.current_weights[f] = (self.current_weights[f] / total_other_weight) * remaining_weight
        
        return {
            'type': 'weight_adjustment',
            'pattern': pattern['type'],
            'factor': factor,
            'amount': amount,
            'old_weights': old_weights,
            'new_weights': self.current_weights.copy(),
            'confidence': pattern['confidence']
        }
    
    def _apply_conservative_tuning(self, amount: float, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conservative tuning to reduce overconfidence"""
        
        old_thresholds = self.current_thresholds.copy()
        
        # Lower all thresholds slightly to be more conservative
        for rating in self.current_thresholds:
            for metric in self.current_thresholds[rating]:
                new_value = self.current_thresholds[rating][metric] - amount
                self.current_thresholds[rating][metric] = max(0.3, min(0.95, new_value))
        
        return {
            'type': 'conservative_tuning',
            'pattern': pattern['type'],
            'amount': -amount,
            'old_thresholds': old_thresholds,
            'new_thresholds': self.current_thresholds.copy(),
            'confidence': pattern['confidence']
        }
    
    def _store_tuning_history(self, changes: List[Dict[str, Any]]):
        """Store algorithm tuning history"""
        
        if not self.learning_sys.is_connected():
            return
        
        try:
            tuning_doc = {
                'timestamp': datetime.utcnow(),
                'type': 'algorithm_tuning',
                'changes': changes,
                'weights_after': self.current_weights.copy(),
                'thresholds_after': self.current_thresholds.copy()
            }
            
            tuning_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}algorithm_tuning"]
            tuning_collection.insert_one(tuning_doc)
            
        except Exception as e:
            print(f"⚠️  Error storing tuning history: {e}")
    
    def get_current_algorithm_config(self) -> Dict[str, Any]:
        """Get current algorithm configuration"""
        return {
            'weights': self.current_weights.copy(),
            'thresholds': self.current_thresholds.copy(),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def test_algorithm_accuracy(self) -> float:
        """Test current algorithm accuracy against recent feedback"""
        
        if not self.learning_sys.is_connected():
            return 0.0
        
        try:
            # Get recent sessions with both auto and user feedback
            feedback_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}user_feedback"]
            
            # Find sessions with both auto and user ratings
            pipeline = [
                {"$group": {
                    "_id": "$session_id",
                    "ratings": {"$push": {"rating": "$rating", "feedback_id": "$feedback_id"}}
                }},
                {"$match": {"ratings.1": {"$exists": True}}}  # At least 2 ratings
            ]
            
            sessions_with_both = list(feedback_collection.aggregate(pipeline))
            
            if not sessions_with_both:
                return 0.0
            
            accurate_predictions = 0
            total_predictions = 0
            
            for session in sessions_with_both:
                ratings = session['ratings']
                auto_rating = None
                user_rating = None
                
                for rating_doc in ratings:
                    if rating_doc['feedback_id'].startswith('auto_'):
                        auto_rating = rating_doc['rating']
                    else:
                        user_rating = rating_doc['rating']
                
                if auto_rating is not None and user_rating is not None:
                    total_predictions += 1
                    # Consider accurate if within 1 star
                    if abs(auto_rating - user_rating) <= 1:
                        accurate_predictions += 1
            
            return accurate_predictions / total_predictions if total_predictions > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️  Error testing algorithm accuracy: {e}")
            return 0.0


# Global instance
algorithm_tuner = AlgorithmTuner()


def auto_tune_algorithm():
    """Main function to automatically tune the algorithm based on feedback"""
    
    print("\n" + "="*60)
    print("Algorithm Auto-Tuning Started")
    print("="*60)
    
    # Analyze feedback patterns
    patterns = algorithm_tuner.analyze_feedback_patterns()
    
    if not patterns:
        print("⚠️  No significant patterns found for tuning")
        return {'applied': 0, 'message': 'No patterns found'}
    
    print(f"✅ Found {len(patterns)} feedback patterns")
    
    for i, pattern in enumerate(patterns, 1):
        print(f"   {i}. {pattern['description']}")
        print(f"      Confidence: {pattern['confidence']:.0%}")
        print(f"      Sample size: {pattern['sample_size']}")
    
    # Test current accuracy
    current_accuracy = algorithm_tuner.test_algorithm_accuracy()
    print(f"\n📊 Current algorithm accuracy: {current_accuracy:.1%}")
    
    # Apply improvements
    results = algorithm_tuner.apply_algorithm_improvements(patterns)
    
    if results['applied'] > 0:
        print(f"\n✅ Applied {results['applied']} algorithm improvements:")
        
        for change in results['changes']:
            print(f"   - {change['type']}: {change.get('pattern', 'unknown')}")
            if 'factor' in change:
                print(f"     Factor: {change['factor']}")
            if 'amount' in change:
                print(f"     Adjustment: {change['amount']:+.3f}")
            print(f"     Confidence: {change['confidence']:.0%}")
        
        # Test new accuracy
        new_accuracy = algorithm_tuner.test_algorithm_accuracy()
        print(f"\n📈 New algorithm accuracy: {new_accuracy:.1%}")
        
        if new_accuracy > current_accuracy:
            print(f"🎉 Improvement: +{(new_accuracy - current_accuracy)*100:.1f}%")
        else:
            print(f"⚠️  Accuracy unchanged or decreased")
        
        return {
            'applied': results['applied'],
            'changes': results['changes'],
            'accuracy_before': current_accuracy,
            'accuracy_after': new_accuracy,
            'improvement': new_accuracy - current_accuracy
        }
    else:
        print("⚠️  No changes applied (insufficient confidence)")
        return {'applied': 0, 'message': 'Insufficient confidence for changes'}


if __name__ == "__main__":
    auto_tune_algorithm()