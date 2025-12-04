"""
optimization_detector.py
------------------------
Detects performance improvements by comparing recent sessions to baseline
Identifies when parameter changes lead to better quality or speed
"""

from learning_system import get_learning_system
from learning_models import SystemOptimization
import uuid
from datetime import datetime, timedelta
import statistics


class OptimizationDetector:
    """Detects performance improvements automatically"""
    
    def __init__(self):
        self.learning_sys = get_learning_system()
    
    def detect_optimizations(self, recent_count=10, baseline_days_start=30, baseline_days_end=60):
        """
        Main detection function
        
        Args:
            recent_count: Number of recent sessions to analyze
            baseline_days_start: Start of baseline period (days ago)
            baseline_days_end: End of baseline period (days ago)
            
        Returns:
            List of detected optimizations
        """
        print("\n" + "="*60)
        print("Optimization Detection Started")
        print("="*60)
        
        # Calculate baseline
        baseline = self._calculate_baseline(baseline_days_start, baseline_days_end)
        
        if not baseline:
            print("⚠️  Not enough baseline data")
            return []
        
        print(f"✅ Baseline calculated from {baseline['sample_size']} sessions")
        print(f"   Avg quality: {baseline['avg_quality']:.2f}")
        print(f"   Avg time: {baseline['avg_time']:.2f}s")
        
        # Get recent sessions
        recent = self._get_recent_sessions(recent_count)
        
        if len(recent) < 5:
            print(f"⚠️  Not enough recent sessions ({len(recent)}/5)")
            return []
        
        print(f"✅ Analyzing {len(recent)} recent sessions")
        
        # Compare to baseline
        optimizations = self._compare_to_baseline(baseline, recent)
        
        print(f"\n📊 Detected {len(optimizations)} optimizations")
        
        return optimizations
    
    def _calculate_baseline(self, days_start, days_end):
        """
        Calculate baseline metrics from historical sessions
        
        Uses sessions from 30-60 days ago to establish baseline
        """
        if not self.learning_sys.is_connected():
            return None
        
        end_date = datetime.utcnow() - timedelta(days=days_start)
        start_date = datetime.utcnow() - timedelta(days=days_end)
        
        sessions = list(self.learning_sys.collections['sessions'].find({
            'status': 'success',
            'timestamp': {'$gte': start_date, '$lte': end_date}
        }))
        
        if len(sessions) < 10:
            # Fallback: use any old sessions if not enough in date range
            sessions = list(self.learning_sys.collections['sessions'].find({
                'status': 'success'
            }).sort('timestamp', 1).limit(20))
            
            if len(sessions) < 10:
                return None
        
        return {
            'avg_quality': statistics.mean([s['extraction_quality_score'] for s in sessions]),
            'avg_time': statistics.mean([s['total_processing_time'] for s in sessions]),
            'avg_entries': statistics.mean([s['final_entries_count'] for s in sessions]),
            'sample_size': len(sessions)
        }
    
    def _get_recent_sessions(self, count):
        """Get most recent successful sessions"""
        if not self.learning_sys.is_connected():
            return []
        
        return list(self.learning_sys.collections['sessions'].find({
            'status': 'success'
        }).sort('timestamp', -1).limit(count))
    
    def _compare_to_baseline(self, baseline, recent):
        """
        Compare recent sessions to baseline and detect improvements
        
        Detects:
        - Quality improvement > 5%
        - Speed improvement > 10%
        """
        # Calculate recent metrics
        recent_quality = statistics.mean([s['extraction_quality_score'] for s in recent])
        recent_time = statistics.mean([s['total_processing_time'] for s in recent])
        recent_entries = statistics.mean([s['final_entries_count'] for s in recent])
        
        # Calculate improvements
        quality_improvement = ((recent_quality - baseline['avg_quality']) / baseline['avg_quality']) * 100
        time_improvement = ((baseline['avg_time'] - recent_time) / baseline['avg_time']) * 100
        
        print(f"\n📈 Comparison to baseline:")
        print(f"   Quality: {quality_improvement:+.1f}%")
        print(f"   Speed: {time_improvement:+.1f}%")
        
        optimizations = []
        
        # Quality improvement detected
        if quality_improvement > 5:
            opt = self._create_optimization(
                "quality",
                f"Quality improved by {quality_improvement:.1f}%",
                quality_improvement,
                0,
                baseline,
                {
                    'avg_quality': round(recent_quality, 2),
                    'avg_time': round(recent_time, 2),
                    'avg_entries': round(recent_entries, 2)
                },
                len(recent)
            )
            optimizations.append(opt)
            print(f"✅ Quality optimization detected: +{quality_improvement:.1f}%")
        
        # Speed improvement detected
        if time_improvement > 10:
            opt = self._create_optimization(
                "speed",
                f"Processing speed improved by {time_improvement:.1f}%",
                0,
                time_improvement,
                baseline,
                {
                    'avg_quality': round(recent_quality, 2),
                    'avg_time': round(recent_time, 2),
                    'avg_entries': round(recent_entries, 2)
                },
                len(recent)
            )
            optimizations.append(opt)
            print(f"✅ Speed optimization detected: +{time_improvement:.1f}%")
        
        # Combined improvement
        if quality_improvement > 3 and time_improvement > 5:
            opt = self._create_optimization(
                "combined",
                f"Quality +{quality_improvement:.1f}%, Speed +{time_improvement:.1f}%",
                quality_improvement,
                time_improvement,
                baseline,
                {
                    'avg_quality': round(recent_quality, 2),
                    'avg_time': round(recent_time, 2),
                    'avg_entries': round(recent_entries, 2)
                },
                len(recent)
            )
            optimizations.append(opt)
            print(f"✅ Combined optimization detected")
        
        return optimizations
    
    def _create_optimization(self, opt_type, description, quality_imp, speed_imp, 
                            old_params, new_params, sample_size):
        """Create optimization record"""
        
        # Calculate confidence based on improvement magnitude and sample size
        improvement_magnitude = max(abs(quality_imp), abs(speed_imp))
        magnitude_factor = min(improvement_magnitude / 20, 1.0) * 0.6
        sample_factor = min(sample_size / 10, 1.0) * 0.4
        confidence = magnitude_factor + sample_factor
        
        return SystemOptimization(
            optimization_id=f"opt_{uuid.uuid4().hex[:16]}",
            optimization_type=opt_type,
            description=description,
            old_parameters=old_params,
            new_parameters=new_params,
            performance_improvement=round(speed_imp, 2),
            quality_improvement=round(quality_imp, 2),
            sample_size=sample_size,
            is_active=True,
            confidence_score=round(confidence, 2)
        )


def detect_optimizations():
    """
    Main function to detect optimizations
    
    Returns:
        Number of optimizations detected
    """
    detector = OptimizationDetector()
    optimizations = detector.detect_optimizations()
    
    learning_sys = get_learning_system()
    
    if not learning_sys.is_connected():
        print("❌ Learning system not connected")
        return 0
    
    stored = 0
    
    for opt in optimizations:
        if learning_sys.store_optimization(opt):
            stored += 1
    
    print(f"\n✅ Stored {stored} optimizations in MongoDB")
    
    return stored


if __name__ == "__main__":
    detect_optimizations()
