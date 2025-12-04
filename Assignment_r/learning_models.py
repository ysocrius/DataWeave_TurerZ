"""
learning_models.py
------------------
Data models for MongoDB Atlas learning system
Defines schemas for processing sessions, performance metrics, and learned patterns
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Document type classification"""
    RESUME = "resume"
    ASSIGNMENT = "assignment"
    EVENT = "event"
    TECHNICAL = "technical"
    BUSINESS = "business"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ChunkResult(BaseModel):
    """Individual chunk processing result"""
    chunk_id: int
    char_range: Optional[str] = None
    page_range: Optional[str] = None
    entries_count: int
    processing_time: float
    content_length: int
    status: str
    error: Optional[str] = None


class DeduplicationStats(BaseModel):
    """Deduplication statistics"""
    initial_count: int
    after_exact_dedup: int
    after_fuzzy_dedup: int
    after_consolidation: int
    final_count: int
    exact_duplicates_removed: int
    fuzzy_duplicates_removed: int
    entries_consolidated: int
    truncation_repairs: int


class ProcessingSession(BaseModel):
    """Complete processing session data"""
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Document metadata
    filename: str
    file_size: int
    total_pages: int
    document_type: DocumentType = DocumentType.UNKNOWN
    
    # Processing parameters
    chunk_size: int
    chunk_overlap: int
    llm_model: str
    temperature: float
    fuzzy_threshold: float
    
    # Processing results
    total_chunks: int
    chunk_results: List[ChunkResult]
    dedup_stats: DeduplicationStats
    final_entries_count: int
    
    # Performance metrics
    total_processing_time: float
    avg_chunk_time: float
    extraction_quality_score: Optional[float] = None
    
    # Status
    status: ProcessingStatus
    error_message: Optional[str] = None
    
    # User feedback (populated later)
    user_rating: Optional[int] = None  # 1-5 stars
    user_corrections_count: Optional[int] = None
    user_feedback_text: Optional[str] = None


class LearnedPattern(BaseModel):
    """Successful parameter combination for specific document type"""
    pattern_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Document characteristics
    document_type: DocumentType
    avg_page_count: int
    avg_file_size: int
    
    # Optimal parameters
    optimal_chunk_size: int
    optimal_chunk_overlap: int
    optimal_fuzzy_threshold: float
    optimal_temperature: float
    
    # Performance metrics
    avg_processing_time: float
    avg_quality_score: float
    success_rate: float
    sample_size: int
    
    # Confidence
    confidence_score: float  # 0.0-1.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class PerformanceMetric(BaseModel):
    """Time-series performance data"""
    metric_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Session reference
    session_id: str
    
    # Performance data
    processing_time: float
    entries_extracted: int
    quality_score: Optional[float] = None
    user_satisfaction: Optional[int] = None
    
    # System metrics
    chunk_count: int
    deduplication_efficiency: float  # % reduction
    error_count: int


class UserFeedback(BaseModel):
    """User feedback and corrections"""
    feedback_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Session reference
    session_id: str
    
    # Feedback data
    rating: int  # 1-5 stars
    feedback_text: Optional[str] = None
    
    # Corrections
    corrections: List[Dict[str, Any]] = []
    corrections_count: int = 0
    
    # Improvement suggestions
    suggested_parameters: Optional[Dict[str, Any]] = None


class SystemOptimization(BaseModel):
    """Auto-discovered system optimizations"""
    optimization_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Optimization details
    optimization_type: str  # "chunking", "deduplication", "prompt", etc.
    description: str
    
    # Parameters
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    
    # Impact
    performance_improvement: float  # % improvement
    quality_improvement: float  # % improvement
    sample_size: int
    
    # Status
    is_active: bool = True
    confidence_score: float


class DocumentClassification(BaseModel):
    """Document type classification result"""
    classification_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Document info
    filename: str
    file_size: int
    page_count: int
    
    # Classification
    document_type: DocumentType
    confidence: float  # 0.0-1.0
    
    # Features used for classification
    features: Dict[str, Any]
    
    # Validation
    user_confirmed: Optional[bool] = None
    actual_type: Optional[DocumentType] = None


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Convert Pydantic model to dictionary for MongoDB"""
    # Use python mode to keep datetime objects as datetime (not strings)
    # This is important for MongoDB date queries
    return model.model_dump(mode='python')


def dict_to_model(data: Dict[str, Any], model_class: type[BaseModel]) -> BaseModel:
    """Convert MongoDB dictionary to Pydantic model"""
    return model_class(**data)
