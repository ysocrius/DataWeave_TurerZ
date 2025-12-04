"""
pattern_analyzer.py
-------------------
Analyzes processing sessions to discover optimal parameter patterns
Groups sessions by characteristics and identifies best-performing configurations
"""

from learning_system import get_learning_system
from learning_models import LearnedPattern, DocumentType
import uuid
from datetime import datetime
from collections import defaultdict
import statistics


class PatternAnalyzer:
    """Analyzes sessions and discovers optimal parameter patterns"""
    
    def __init__(self):
        self.learning_sys = get_learning_system()
    
    def analyze_and_learn(self, min_sessions=10, min_group_size=5):
        """
        Main analysis function
        
        Args:
            min_sessions: Minimum total sessions required
            min_group_size: Minimum sessions per group to create pattern
            
        Returns:
            List of created patterns
        """
        print("\n" + "="*60)
        print("Pattern Analysis Started")
        print("="*60)
        
        # Get successful sessions
        sessions = self._get_successful_sessions()
        
        if len(sessions) < min_sessions:
            print(f"⚠️  Not enough sessions ({len(sessions)}/{min_sessions})")
            return []
        
        print(f"✅ Analyzing {len(sessions)} successful sessions")
        
        # Group by characteristics
        groups = self._group_sessions(sessions)
        print(f"✅ Created {len(groups)} groups")
        
        # Create patterns for each group
        patterns = []
        for group_key, group_sessions in groups.items():
            if len(group_sessions) >= min_group_size:
                pattern = self._create_pattern(group_key, group_sessions)
                if pattern:
                    patterns.append(pattern)
                    print(f"✅ Pattern: {group_key} (n={len(group_sessions)}, conf={pattern.confidence_score:.2f})")
            else:
                print(f"⚠️  Skipped: {group_key} (only {len(group_sessions)} sessions)")
        
        print(f"\n📊 Created {len(patterns)} patterns")
        return patterns
    
    def _get_successful_sessions(self, limit=100):
        """Get successful sessions with quality > 0.7"""
        if not self.learning_sys.is_connected():
            return []
        
        return list(self.learning_sys.collections['sessions'].find({
            'status': 'success',
            'extraction_quality_score': {'$gte': 0.7}
        }).sort('timestamp', -1).limit(limit))
    
    def _group_sessions(self, sessions):
        """
        Group sessions by document characteristics
        
        Groups by:
        - Page count buckets: 1-3, 4-6, 7-10, 10+
        - File size buckets: small (<100KB), medium (100-500KB), large (>500KB)
        """
        groups = defaultdict(list)
        
        for session in sessions:
            # Page count buckets
            pages = session['total_pages']
            if pages <= 3:
                page_bucket = '1-3'
            elif pages <= 6:
                page_bucket = '4-6'
            elif pages <= 10:
                page_bucket = '7-10'
            else:
                page_bucket = '10+'
            
            # File size buckets
            size = session['file_size']
            if size < 100000:
                size_bucket = 'small'
            elif size < 500000:
                size_bucket = 'medium'
            else:
                size_bucket = 'large'
            
            group_key = f"pages_{page_bucket}_size_{size_bucket}"
            groups[group_key].append(session)
        
        return groups
    
    def _create_pattern(self, group_key, sessions):
        """
        Create pattern from group of sessions
        
        Finds the parameter combination with highest quality score
        and calculates confidence based on sample size and consistency
        """
        # Extract parameters and metrics
        chunk_sizes = [s['chunk_size'] for s in sessions]
        overlaps = [s['chunk_overlap'] for s in sessions]
        thresholds = [s['fuzzy_threshold'] for s in sessions]
        qualities = [s['extraction_quality_score'] for s in sessions]
        times = [s['total_processing_time'] for s in sessions]
        pages = [s['total_pages'] for s in sessions]
        sizes = [s['file_size'] for s in sessions]
        
        # Find parameters with highest quality
        best_quality_idx = qualities.index(max(qualities))
        
        # Calculate confidence
        confidence = self._calculate_confidence(sessions, qualities)
        
        if confidence < 0.6:
            return None
        
        # Create pattern
        pattern = LearnedPattern(
            pattern_id=f"pattern_{uuid.uuid4().hex[:16]}",
            document_type=DocumentType.UNKNOWN,
            avg_page_count=int(statistics.mean(pages)),
            avg_file_size=int(statistics.mean(sizes)),
            optimal_chunk_size=chunk_sizes[best_quality_idx],
            optimal_chunk_overlap=overlaps[best_quality_idx],
            optimal_fuzzy_threshold=thresholds[best_quality_idx],
            optimal_temperature=0.0,
            avg_processing_time=round(statistics.mean(times), 2),
            avg_quality_score=round(statistics.mean(qualities), 2),
            success_rate=100.0,
            sample_size=len(sessions),
            confidence_score=round(confidence, 2)
        )
        
        return pattern
    
    def _calculate_confidence(self, sessions, qualities):
        """
        Calculate pattern confidence score (0.0-1.0)
        
        Based on:
        - Sample size (40%): More sessions = higher confidence
        - Quality consistency (30%): Lower variance = higher confidence
        - Success rate (30%): All successful = higher confidence
        """
        # Sample size factor (40%)
        # Reaches 1.0 at 20 sessions
        sample_factor = min(len(sessions) / 20, 1.0) * 0.4
        
        # Quality consistency factor (30%)
        if len(qualities) > 1:
            std_dev = statistics.stdev(qualities)
            mean_quality = statistics.mean(qualities)
            if mean_quality > 0:
                consistency = 1.0 - (std_dev / mean_quality)
                consistency_factor = max(0, min(consistency, 1.0)) * 0.3
            else:
                consistency_factor = 0.0
        else:
            consistency_factor = 0.3
        
        # Success rate factor (30%)
        # All sessions are successful (filtered earlier)
        success_factor = 1.0 * 0.3
        
        total_confidence = sample_factor + consistency_factor + success_factor
        
        return min(total_confidence, 1.0)


def learn_patterns():
    """
    Main function to learn patterns from existing sessions
    
    Returns:
        Number of patterns created
    """
    analyzer = PatternAnalyzer()
    patterns = analyzer.analyze_and_learn()
    
    learning_sys = get_learning_system()
    
    if not learning_sys.is_connected():
        print("❌ Learning system not connected")
        return 0
    
    stored = 0
    
    for pattern in patterns:
        if learning_sys.store_learned_pattern(pattern):
            stored += 1
    
    print(f"\n✅ Stored {stored} patterns in MongoDB")
    
    return stored


if __name__ == "__main__":
    learn_patterns()
