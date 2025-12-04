"""
learning_system.py
------------------
Core MongoDB Atlas learning system infrastructure
Handles connection, data storage, and retrieval for the self-improving AI system
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
from config import Config
from learning_models import (
    ProcessingSession, LearnedPattern, PerformanceMetric,
    UserFeedback, SystemOptimization, DocumentClassification,
    model_to_dict, dict_to_model
)


class MongoLearningSystem:
    """MongoDB Atlas learning system for AI document processor"""
    
    def __init__(self):
        """Initialize MongoDB connection and collections"""
        self.client = None
        self.db = None
        self.collections = {}
        self._connected = False
        
        # Only initialize if learning is enabled
        if Config.LEARNING_ENABLED:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize MongoDB Atlas connection"""
        if not Config.MONGODB_ATLAS_URI:
            print("⚠️  MongoDB Atlas URI not configured. Learning system disabled.")
            return
        
        try:
            # Create MongoDB client with timeout
            self.client = MongoClient(
                Config.MONGODB_ATLAS_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                retryWrites=True
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database
            self.db = self.client[Config.MONGODB_DATABASE_NAME]
            
            # Initialize collections
            prefix = Config.MONGODB_COLLECTION_PREFIX
            self.collections = {
                'sessions': self.db[f'{prefix}processing_sessions'],
                'patterns': self.db[f'{prefix}learned_patterns'],
                'metrics': self.db[f'{prefix}performance_metrics'],
                'feedback': self.db[f'{prefix}user_feedback'],
                'optimizations': self.db[f'{prefix}system_optimizations'],
                'classifications': self.db[f'{prefix}document_classifications']
            }
            
            # Create indexes
            self._create_indexes()
            
            self._connected = True
            print("✅ MongoDB Atlas learning system connected")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"⚠️  MongoDB connection failed: {e}")
            print("   Learning system will be disabled for this session")
            self._connected = False
        except Exception as e:
            print(f"⚠️  Unexpected error initializing learning system: {e}")
            self._connected = False
    
    def _create_indexes(self):
        """Create performance indexes for collections"""
        try:
            # Processing sessions indexes
            self.collections['sessions'].create_index([('session_id', ASCENDING)], unique=True)
            self.collections['sessions'].create_index([('timestamp', DESCENDING)])
            self.collections['sessions'].create_index([('document_type', ASCENDING)])
            self.collections['sessions'].create_index([('status', ASCENDING)])
            
            # Learned patterns indexes
            self.collections['patterns'].create_index([('pattern_id', ASCENDING)], unique=True)
            self.collections['patterns'].create_index([('document_type', ASCENDING)])
            self.collections['patterns'].create_index([('confidence_score', DESCENDING)])
            self.collections['patterns'].create_index([('last_updated', DESCENDING)])
            
            # Performance metrics indexes
            self.collections['metrics'].create_index([('metric_id', ASCENDING)], unique=True)
            self.collections['metrics'].create_index([('session_id', ASCENDING)])
            self.collections['metrics'].create_index([('timestamp', DESCENDING)])
            
            # User feedback indexes
            self.collections['feedback'].create_index([('feedback_id', ASCENDING)], unique=True)
            self.collections['feedback'].create_index([('session_id', ASCENDING)])
            self.collections['feedback'].create_index([('rating', DESCENDING)])
            
            # System optimizations indexes
            self.collections['optimizations'].create_index([('optimization_id', ASCENDING)], unique=True)
            self.collections['optimizations'].create_index([('optimization_type', ASCENDING)])
            self.collections['optimizations'].create_index([('is_active', ASCENDING)])
            
            # Document classifications indexes
            self.collections['classifications'].create_index([('classification_id', ASCENDING)], unique=True)
            self.collections['classifications'].create_index([('document_type', ASCENDING)])
            self.collections['classifications'].create_index([('confidence', DESCENDING)])
            
            # TTL index for old metrics (optional - keep 90 days)
            self.collections['metrics'].create_index(
                [('timestamp', ASCENDING)],
                expireAfterSeconds=90*24*60*60  # 90 days
            )
            
        except Exception as e:
            print(f"⚠️  Error creating indexes: {e}")
    
    def is_connected(self) -> bool:
        """Check if learning system is connected"""
        return self._connected
    
    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        if not self._connected:
            return False
        
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            return False
    
    # ==================== Processing Sessions ====================
    
    def store_processing_session(self, session: ProcessingSession) -> bool:
        """Store a processing session"""
        if not self._connected:
            return False
        
        try:
            session_dict = model_to_dict(session)
            self.collections['sessions'].insert_one(session_dict)
            return True
        except Exception as e:
            print(f"⚠️  Error storing processing session: {e}")
            return False
    
    def get_processing_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Retrieve a processing session by ID"""
        if not self._connected:
            return None
        
        try:
            data = self.collections['sessions'].find_one({'session_id': session_id})
            if data:
                return dict_to_model(data, ProcessingSession)
            return None
        except Exception as e:
            print(f"⚠️  Error retrieving processing session: {e}")
            return None
    
    def get_recent_sessions(self, limit: int = 10) -> List[ProcessingSession]:
        """Get recent processing sessions"""
        if not self._connected:
            return []
        
        try:
            cursor = self.collections['sessions'].find().sort('timestamp', DESCENDING).limit(limit)
            return [dict_to_model(doc, ProcessingSession) for doc in cursor]
        except Exception as e:
            print(f"⚠️  Error retrieving recent sessions: {e}")
            return []
    
    # ==================== Learned Patterns ====================
    
    def store_learned_pattern(self, pattern: LearnedPattern) -> bool:
        """Store or update a learned pattern"""
        if not self._connected:
            return False
        
        try:
            pattern_dict = model_to_dict(pattern)
            self.collections['patterns'].update_one(
                {'pattern_id': pattern.pattern_id},
                {'$set': pattern_dict},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"⚠️  Error storing learned pattern: {e}")
            return False
    
    def get_optimal_parameters(self, document_type: str, page_count: int, file_size: int) -> Optional[Dict[str, Any]]:
        """Get optimal parameters for a document type"""
        if not self._connected:
            return None
        
        try:
            # Find patterns for this document type
            patterns = list(self.collections['patterns'].find({
                'document_type': document_type,
                'confidence_score': {'$gte': 0.7}  # Only high-confidence patterns
            }).sort('confidence_score', DESCENDING).limit(5))
            
            if not patterns:
                return None
            
            # Find best matching pattern based on page count and file size
            best_pattern = None
            best_score = 0
            
            for pattern in patterns:
                # Calculate similarity score
                page_diff = abs(pattern['avg_page_count'] - page_count) / max(page_count, 1)
                size_diff = abs(pattern['avg_file_size'] - file_size) / max(file_size, 1)
                similarity = 1.0 - (page_diff + size_diff) / 2
                
                score = similarity * pattern['confidence_score']
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
            
            if best_pattern:
                return {
                    'chunk_size': best_pattern['optimal_chunk_size'],
                    'chunk_overlap': best_pattern['optimal_chunk_overlap'],
                    'fuzzy_threshold': best_pattern['optimal_fuzzy_threshold'],
                    'temperature': best_pattern['optimal_temperature'],
                    'confidence': best_score
                }
            
            return None
        except Exception as e:
            print(f"⚠️  Error retrieving optimal parameters: {e}")
            return None
    
    # ==================== Performance Metrics ====================
    
    def store_performance_metric(self, metric: PerformanceMetric) -> bool:
        """Store a performance metric"""
        if not self._connected:
            return False
        
        try:
            metric_dict = model_to_dict(metric)
            self.collections['metrics'].insert_one(metric_dict)
            return True
        except Exception as e:
            print(f"⚠️  Error storing performance metric: {e}")
            return False
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over time"""
        if not self._connected:
            return {}
        
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Try metrics collection first
            pipeline = [
                {'$match': {'timestamp': {'$gte': start_date}}},
                {'$group': {
                    '_id': None,
                    'avg_processing_time': {'$avg': '$processing_time'},
                    'avg_entries': {'$avg': '$entries_extracted'},
                    'avg_quality': {'$avg': '$quality_score'},
                    'total_sessions': {'$sum': 1}
                }}
            ]
            
            result = list(self.collections['metrics'].aggregate(pipeline))
            if result and result[0].get('total_sessions', 0) > 0:
                return result[0]
            
            # Fallback to sessions collection if no metrics found
            session_pipeline = [
                {'$match': {'timestamp': {'$gte': start_date}}},
                {'$group': {
                    '_id': None,
                    'avg_processing_time': {'$avg': '$total_processing_time'},
                    'avg_entries': {'$avg': '$final_entries_count'},
                    'avg_quality': {'$avg': '$extraction_quality_score'},
                    'total_sessions': {'$sum': 1}
                }}
            ]
            
            session_result = list(self.collections['sessions'].aggregate(session_pipeline))
            if session_result:
                return session_result[0]
            
            return {}
        except Exception as e:
            print(f"⚠️  Error retrieving performance trends: {e}")
            return {}
    
    # ==================== User Feedback ====================
    
    def store_user_feedback(self, feedback: UserFeedback) -> bool:
        """Store user feedback"""
        if not self._connected:
            return False
        
        try:
            feedback_dict = model_to_dict(feedback)
            self.collections['feedback'].insert_one(feedback_dict)
            
            # Update session with feedback
            self.collections['sessions'].update_one(
                {'session_id': feedback.session_id},
                {'$set': {
                    'user_rating': feedback.rating,
                    'user_corrections_count': feedback.corrections_count,
                    'user_feedback_text': feedback.feedback_text
                }}
            )
            
            return True
        except Exception as e:
            print(f"⚠️  Error storing user feedback: {e}")
            return False
    
    # ==================== System Optimizations ====================
    
    def store_optimization(self, optimization: SystemOptimization) -> bool:
        """Store a system optimization"""
        if not self._connected:
            return False
        
        try:
            optimization_dict = model_to_dict(optimization)
            self.collections['optimizations'].insert_one(optimization_dict)
            return True
        except Exception as e:
            print(f"⚠️  Error storing optimization: {e}")
            return False
    
    def get_active_optimizations(self, optimization_type: Optional[str] = None) -> List[SystemOptimization]:
        """Get active system optimizations"""
        if not self._connected:
            return []
        
        try:
            query = {'is_active': True}
            if optimization_type:
                query['optimization_type'] = optimization_type
            
            cursor = self.collections['optimizations'].find(query).sort('confidence_score', DESCENDING)
            return [dict_to_model(doc, SystemOptimization) for doc in cursor]
        except Exception as e:
            print(f"⚠️  Error retrieving optimizations: {e}")
            return []
    
    # ==================== Document Classification ====================
    
    def store_classification(self, classification: DocumentClassification) -> bool:
        """Store a document classification"""
        if not self._connected:
            return False
        
        try:
            classification_dict = model_to_dict(classification)
            self.collections['classifications'].insert_one(classification_dict)
            return True
        except Exception as e:
            print(f"⚠️  Error storing classification: {e}")
            return False
    
    # ==================== Analytics ====================
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics with separate auto/user ratings"""
        if not self._connected:
            return {}
        
        try:
            analytics = {
                'total_sessions': self.collections['sessions'].count_documents({}),
                'total_patterns': self.collections['patterns'].count_documents({}),
                'total_feedback': self.collections['feedback'].count_documents({}),
                'active_optimizations': self.collections['optimizations'].count_documents({'is_active': True}),
                'success_rate': 0
            }
            
            # Separate automated vs user feedback
            auto_feedback_count = self.collections['feedback'].count_documents({
                'feedback_id': {'$regex': '^auto_'}
            })
            user_feedback_count = self.collections['feedback'].count_documents({
                'feedback_id': {'$not': {'$regex': '^auto_'}}
            })
            
            # Calculate average automated rating
            auto_rating_pipeline = [
                {'$match': {'feedback_id': {'$regex': '^auto_'}}},
                {'$group': {'_id': None, 'avg_rating': {'$avg': '$rating'}}}
            ]
            auto_rating_result = list(self.collections['feedback'].aggregate(auto_rating_pipeline))
            avg_auto_rating = round(auto_rating_result[0]['avg_rating'], 2) if auto_rating_result else 0
            
            # Calculate average user rating (manual/overrides)
            user_rating_pipeline = [
                {'$match': {'feedback_id': {'$not': {'$regex': '^auto_'}}}},
                {'$group': {'_id': None, 'avg_rating': {'$avg': '$rating'}}}
            ]
            user_rating_result = list(self.collections['feedback'].aggregate(user_rating_pipeline))
            avg_user_rating = round(user_rating_result[0]['avg_rating'], 2) if user_rating_result else 0
            
            # Calculate combined average
            combined_rating_pipeline = [
                {'$group': {'_id': None, 'avg_rating': {'$avg': '$rating'}}}
            ]
            combined_rating_result = list(self.collections['feedback'].aggregate(combined_rating_pipeline))
            avg_combined_rating = round(combined_rating_result[0]['avg_rating'], 2) if combined_rating_result else 0
            
            # Calculate override rate
            override_rate = 0
            agreement_rate = 0
            if analytics['total_sessions'] > 0:
                override_rate = round((user_feedback_count / analytics['total_sessions']) * 100, 1)
                agreement_rate = round(((analytics['total_sessions'] - user_feedback_count) / analytics['total_sessions']) * 100, 1)
            
            # Add rating breakdown
            analytics['feedback_breakdown'] = {
                'auto_feedback_count': auto_feedback_count,
                'user_feedback_count': user_feedback_count,
                'avg_auto_rating': avg_auto_rating,
                'avg_user_rating': avg_user_rating,
                'avg_combined_rating': avg_combined_rating,
                'override_rate': override_rate,
                'agreement_rate': agreement_rate
            }
            
            # Keep legacy field for backward compatibility
            analytics['avg_user_rating'] = avg_combined_rating
            
            # Calculate success rate
            success_pipeline = [
                {'$group': {
                    '_id': '$status',
                    'count': {'$sum': 1}
                }}
            ]
            status_result = list(self.collections['sessions'].aggregate(success_pipeline))
            total = sum(r['count'] for r in status_result)
            success = sum(r['count'] for r in status_result if r['_id'] == 'success')
            if total > 0:
                analytics['success_rate'] = round((success / total) * 100, 2)
            
            return analytics
        except Exception as e:
            print(f"⚠️  Error retrieving analytics: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self._connected = False


# Global learning system instance
learning_system = MongoLearningSystem()


def get_learning_system() -> MongoLearningSystem:
    """Get the global learning system instance"""
    return learning_system
