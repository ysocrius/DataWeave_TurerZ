"""
learning_orchestrator.py
-------------------------
Orchestrates the complete feedback learning system
Coordinates feedback analysis, pattern recognition, and algorithm tuning
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from feedback_analyzer import feedback_analyzer, analyze_user_feedback
from pattern_recognizer import pattern_recognizer, analyze_feedback_patterns
from algorithm_tuner import algorithm_tuner, auto_tune_algorithm
from learning_system import get_learning_system
from config import Config


class LearningOrchestrator:
    """Orchestrates the complete feedback learning pipeline"""
    
    def __init__(self):
        self.learning_sys = get_learning_system()
        self.min_feedback_for_tuning = 10
        self.min_patterns_for_tuning = 2
    
    def run_complete_learning_cycle(self) -> Dict[str, Any]:
        """Run the complete learning cycle: analyze → recognize → tune"""
        
        print("\n" + "="*70)
        print("🧠 COMPLETE FEEDBACK LEARNING CYCLE")
        print("="*70)
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'feedback_analysis': {},
            'pattern_recognition': {},
            'algorithm_tuning': {},
            'overall_success': False,
            'improvements_made': 0
        }
        
        try:
            # Phase 1: Analyze User Feedback
            print("\n🔍 Phase 1: Analyzing User Feedback...")
            feedback_results = analyze_user_feedback()
            results['feedback_analysis'] = feedback_results
            
            if feedback_results.get('total_analyzed', 0) < self.min_feedback_for_tuning:
                print(f"⚠️  Insufficient feedback for tuning ({feedback_results.get('total_analyzed', 0)}/{self.min_feedback_for_tuning})")
                return results
            
            print(f"✅ Analyzed {feedback_results.get('total_analyzed', 0)} feedback comments")
            
            # Phase 2: Recognize Patterns
            print("\n🔍 Phase 2: Recognizing Feedback Patterns...")
            patterns = analyze_feedback_patterns()
            results['pattern_recognition'] = {
                'patterns_found': len(patterns),
                'patterns': patterns
            }
            
            if len(patterns) < self.min_patterns_for_tuning:
                print(f"⚠️  Insufficient patterns for tuning ({len(patterns)}/{self.min_patterns_for_tuning})")
                return results
            
            print(f"✅ Found {len(patterns)} actionable patterns")
            
            # Phase 3: Auto-Tune Algorithm
            print("\n🔧 Phase 3: Auto-Tuning Algorithm...")
            tuning_results = auto_tune_algorithm()
            results['algorithm_tuning'] = tuning_results
            
            if tuning_results.get('applied', 0) > 0:
                print(f"✅ Applied {tuning_results['applied']} algorithm improvements")
                results['improvements_made'] = tuning_results['applied']
                results['overall_success'] = True
                
                # Store learning cycle results
                self._store_learning_cycle_results(results)
                
                print("\n🎉 Learning cycle completed successfully!")
                print(f"   Improvements: {results['improvements_made']}")
                if 'accuracy_before' in tuning_results and 'accuracy_after' in tuning_results:
                    accuracy_change = tuning_results['accuracy_after'] - tuning_results['accuracy_before']
                    print(f"   Accuracy change: {accuracy_change:+.1%}")
            else:
                print("⚠️  No algorithm changes applied")
            
            return results
            
        except Exception as e:
            print(f"❌ Learning cycle failed: {e}")
            results['error'] = str(e)
            return results
    
    def run_feedback_analysis_only(self) -> Dict[str, Any]:
        """Run only feedback analysis (lighter operation)"""
        
        print("\n🔍 Running Feedback Analysis...")
        
        try:
            results = analyze_user_feedback()
            
            if results.get('total_analyzed', 0) > 0:
                print(f"✅ Analyzed {results['total_analyzed']} feedback comments")
                
                if results.get('suggestions'):
                    print(f"💡 Generated {len(results['suggestions'])} improvement suggestions")
                    for i, suggestion in enumerate(results['suggestions'], 1):
                        print(f"   {i}. {suggestion.get('description', '')}")
            else:
                print("⚠️  No feedback to analyze")
            
            return results
            
        except Exception as e:
            print(f"❌ Feedback analysis failed: {e}")
            return {'error': str(e)}
    
    def get_learning_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the learning system"""
        
        if not self.learning_sys.is_connected():
            return {'connected': False, 'error': 'Learning system not connected'}
        
        try:
            # Get basic analytics
            analytics = self.learning_sys.get_system_analytics()
            
            # Get recent learning activity
            recent_analyses = self._get_recent_learning_activity()
            
            # Get algorithm configuration
            current_config = algorithm_tuner.get_current_algorithm_config()
            
            # Calculate learning metrics
            learning_metrics = self._calculate_learning_metrics()
            
            return {
                'connected': True,
                'analytics': analytics,
                'recent_activity': recent_analyses,
                'algorithm_config': current_config,
                'learning_metrics': learning_metrics,
                'status': 'operational'
            }
            
        except Exception as e:
            return {'connected': True, 'error': str(e), 'status': 'error'}
    
    def _get_recent_learning_activity(self) -> Dict[str, Any]:
        """Get recent learning system activity"""
        
        try:
            # Get recent feedback analyses
            analysis_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}feedback_analysis"]
            recent_analyses = list(analysis_collection.find().sort('timestamp', -1).limit(5))
            
            # Get recent algorithm tuning
            tuning_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}algorithm_tuning"]
            recent_tuning = list(tuning_collection.find().sort('timestamp', -1).limit(3))
            
            # Get recent learning cycles
            cycles_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}learning_cycles"]
            recent_cycles = list(cycles_collection.find().sort('timestamp', -1).limit(3))
            
            return {
                'recent_analyses': len(recent_analyses),
                'recent_tuning': len(recent_tuning),
                'recent_cycles': len(recent_cycles),
                'last_analysis': recent_analyses[0]['timestamp'] if recent_analyses else None,
                'last_tuning': recent_tuning[0]['timestamp'] if recent_tuning else None,
                'last_cycle': recent_cycles[0]['timestamp'] if recent_cycles else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_learning_metrics(self) -> Dict[str, Any]:
        """Calculate learning system performance metrics"""
        
        try:
            # Get feedback data
            feedback_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}user_feedback"]
            
            # Calculate override rate trend
            total_feedback = feedback_collection.count_documents({})
            auto_feedback = feedback_collection.count_documents({'feedback_id': {'$regex': '^auto_'}})
            user_feedback = total_feedback - auto_feedback
            
            override_rate = (user_feedback / total_feedback * 100) if total_feedback > 0 else 0
            
            # Calculate accuracy trend (simplified)
            accuracy = algorithm_tuner.test_algorithm_accuracy()
            
            # Get improvement trend
            tuning_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}algorithm_tuning"]
            total_improvements = tuning_collection.count_documents({})
            
            return {
                'override_rate': round(override_rate, 1),
                'algorithm_accuracy': round(accuracy * 100, 1),
                'total_improvements': total_improvements,
                'feedback_coverage': round((total_feedback / max(1, self.learning_sys.get_system_analytics().get('total_sessions', 1))) * 100, 1)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _store_learning_cycle_results(self, results: Dict[str, Any]):
        """Store learning cycle results for tracking"""
        
        if not self.learning_sys.is_connected():
            return
        
        try:
            cycles_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}learning_cycles"]
            cycles_collection.insert_one(results)
            
        except Exception as e:
            print(f"⚠️  Error storing learning cycle results: {e}")
    
    def should_run_learning_cycle(self) -> Dict[str, Any]:
        """Determine if a learning cycle should be run"""
        
        if not self.learning_sys.is_connected():
            return {'should_run': False, 'reason': 'Learning system not connected'}
        
        try:
            # Check if enough new feedback since last cycle
            cycles_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}learning_cycles"]
            last_cycle = cycles_collection.find_one(sort=[('timestamp', -1)])
            
            feedback_collection = self.learning_sys.db[f"{Config.MONGODB_COLLECTION_PREFIX}user_feedback"]
            
            if last_cycle:
                last_cycle_time = last_cycle['timestamp']
                new_feedback_count = feedback_collection.count_documents({
                    'timestamp': {'$gt': last_cycle_time},
                    'feedback_id': {'$not': {'$regex': '^auto_'}}
                })
            else:
                new_feedback_count = feedback_collection.count_documents({
                    'feedback_id': {'$not': {'$regex': '^auto_'}}
                })
            
            if new_feedback_count >= self.min_feedback_for_tuning:
                return {
                    'should_run': True,
                    'reason': f'{new_feedback_count} new feedback available',
                    'new_feedback_count': new_feedback_count
                }
            else:
                return {
                    'should_run': False,
                    'reason': f'Only {new_feedback_count} new feedback (need {self.min_feedback_for_tuning})',
                    'new_feedback_count': new_feedback_count
                }
                
        except Exception as e:
            return {'should_run': False, 'reason': f'Error checking: {e}'}


# Global instance
learning_orchestrator = LearningOrchestrator()


def run_smart_learning():
    """Smart learning function that decides what to run based on available data"""
    
    # Check if we should run a full cycle
    should_run = learning_orchestrator.should_run_learning_cycle()
    
    if should_run['should_run']:
        print(f"🚀 Running full learning cycle: {should_run['reason']}")
        return learning_orchestrator.run_complete_learning_cycle()
    else:
        print(f"🔍 Running feedback analysis only: {should_run['reason']}")
        return learning_orchestrator.run_feedback_analysis_only()


if __name__ == "__main__":
    run_smart_learning()