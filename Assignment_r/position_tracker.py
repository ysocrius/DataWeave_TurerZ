"""
position_tracker.py
------------------
Enhanced Method 1: Character Position + Chunk Order tracking for fixing row order issues.
Ensures Excel output rows appear in the same order as information appears in the original PDF.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from fuzzywuzzy import fuzz


class PositionTracker:
    """
    Enhanced position tracking using multiple search strategies
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.method_stats = {
            'multi_candidate': 0,
            'fuzzy_match': 0, 
            'context_aware': 0,
            'intelligent_fallback': 0
        }
    
    def fix_entry_ordering(self, entries: List[Dict], full_text: str, chunks: List[Dict]) -> List[Dict]:
        """
        Main function to fix entry ordering using Enhanced Method 1
        
        Args:
            entries: List of extracted entries with chunk_id
            full_text: Original PDF text
            chunks: List of chunk information
            
        Returns:
            List of entries sorted by document position
        """
        
        # Create chunk lookup for fast access
        chunk_lookup = {i: chunk for i, chunk in enumerate(chunks)}
        
        positioned_entries = []
        
        for entry_idx, entry in enumerate(entries):
            # Add entry index for fallback positioning
            entry['entry_index'] = entry_idx
            
            # Get chunk information
            chunk_id = entry.get('chunk_id', 0)
            chunk_info = chunk_lookup.get(chunk_id, {})
            
            # Find position using multiple strategies
            position, confidence, method = self._detect_position(
                entry, full_text, chunk_info
            )
            
            # Calculate final position with chunk priority
            chunk_priority = chunk_id * 100000  # Large gaps between chunks
            final_position = chunk_priority + position
            
            # Add position metadata to entry
            entry.update({
                'final_position': final_position,
                'position_confidence': confidence,
                'position_method': method,
                'debug_info': {
                    'chunk_id': chunk_id,
                    'local_position': position,
                    'chunk_start': chunk_info.get('start_pos', 0)
                }
            })
            
            positioned_entries.append(entry)
            self.method_stats[method] += 1
        
        # Sort by final position
        sorted_entries = sorted(positioned_entries, key=lambda x: x['final_position'])
        
        # Log statistics
        self._log_statistics(len(entries))
        
        return sorted_entries
    
    def _detect_position(self, entry: Dict, full_text: str, chunk_info: Dict) -> Tuple[int, float, str]:
        """
        Detect entry position using multiple strategies
        
        Returns:
            Tuple of (position, confidence, method_used)
        """
        
        chunk_start = chunk_info.get('start_pos', 0)
        chunk_end = chunk_info.get('end_pos', len(full_text))
        chunk_text = full_text[chunk_start:chunk_end]
        
        # Strategy 1: Multi-candidate search
        position, confidence = self._multi_candidate_search(entry, chunk_text, chunk_start)
        if confidence > 0.7:
            return position, confidence, 'multi_candidate'
        
        # Strategy 2: Fuzzy matching
        position_fuzzy, confidence_fuzzy = self._fuzzy_position_search(entry, chunk_text, chunk_start)
        if confidence_fuzzy > 0.6:
            return position_fuzzy, confidence_fuzzy, 'fuzzy_match'
        
        # Strategy 3: Context-aware search
        position_context, confidence_context = self._context_aware_search(entry, chunk_text, chunk_start)
        if confidence_context > 0.5:
            return position_context, confidence_context, 'context_aware'
        
        # Strategy 4: Intelligent fallback
        position_fallback, confidence_fallback = self._intelligent_fallback(entry, chunk_info)
        return position_fallback, confidence_fallback, 'intelligent_fallback'
    
    def _multi_candidate_search(self, entry: Dict, chunk_text: str, chunk_start: int) -> Tuple[int, float]:
        """
        Try multiple search candidates in order of preference
        """
        
        candidates = self._extract_search_candidates(entry)
        
        for i, candidate in enumerate(candidates):
            if candidate and len(candidate) > 2:
                pos = chunk_text.find(candidate)
                if pos != -1:
                    # Higher confidence for earlier (more specific) candidates
                    confidence = 1.0 - (i * 0.1)
                    return chunk_start + pos, max(0.3, confidence)
        
        return 0, 0.0
    
    def _fuzzy_position_search(self, entry: Dict, chunk_text: str, chunk_start: int) -> Tuple[int, float]:
        """
        Use fuzzy matching when exact search fails
        """
        
        search_text = f"{entry.get('key') or ''} {entry.get('value') or ''}"[:50]
        if not search_text.strip():
            return 0, 0.0
        
        # Split chunk into lines for better matching
        lines = chunk_text.split('\n')
        
        best_position = 0
        best_score = 0
        current_pos = 0
        
        for line in lines:
            if line.strip():
                score = fuzz.partial_ratio(search_text.lower(), line.lower())
                if score > best_score and score > 70:  # 70% similarity threshold
                    best_score = score
                    best_position = current_pos
            
            current_pos += len(line) + 1  # +1 for newline
        
        if best_score > 70:
            confidence = best_score / 100.0
            return chunk_start + best_position, confidence
        
        return 0, 0.0
    
    def _context_aware_search(self, entry: Dict, chunk_text: str, chunk_start: int) -> Tuple[int, float]:
        """
        Use regex patterns for structured data
        """
        
        key = entry.get('key') or ''
        value = entry.get('value') or ''
        
        if not key or not value:
            return 0, 0.0
        
        # Common patterns for structured data
        patterns = [
            # Pattern 1: Key: Value
            rf"{re.escape(key)}\s*:\s*{re.escape(value[:20])}",
            # Pattern 2: Key = Value  
            rf"{re.escape(key)}\s*=\s*{re.escape(value[:20])}",
            # Pattern 3: Key with partial value
            rf"{re.escape(key)}\s*[:=]\s*.*{re.escape(value.split()[0] if value.split() else '')}",
            # Pattern 4: Phone number pattern
            rf"\+?1?\s*\(?\d{{3}}\)?\s*[-.\s]?\d{{3}}\s*[-.\s]?\d{{4}}" if 'phone' in key.lower() else None,
            # Pattern 5: Email pattern
            rf"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}}" if 'email' in key.lower() else None
        ]
        
        for pattern in patterns:
            if pattern:
                try:
                    match = re.search(pattern, chunk_text, re.IGNORECASE)
                    if match:
                        return chunk_start + match.start(), 0.9
                except re.error:
                    continue  # Skip invalid regex patterns
        
        return 0, 0.0
    
    def _intelligent_fallback(self, entry: Dict, chunk_info: Dict) -> Tuple[int, float]:
        """
        Smart fallback when all searches fail
        """
        
        chunk_start = chunk_info.get('start_pos', 0)
        chunk_length = chunk_info.get('end_pos', chunk_start + 1000) - chunk_start
        entry_index = entry.get('entry_index', 0)
        
        position_hints = []
        
        # Hint 1: Alphabetical order of keys
        key = entry.get('key') or ''
        if key and key[0].isalpha():
            first_char = key[0].lower()
            alpha_position = (ord(first_char) - ord('a')) / 26.0
            position_hints.append(alpha_position)
        
        # Hint 2: Common field ordering patterns
        key_lower = key.lower()
        if any(field in key_lower for field in ['name', 'title', 'id']):
            position_hints.append(0.1)  # Early in document
        elif any(field in key_lower for field in ['notes', 'comments', 'remarks']):
            position_hints.append(0.9)  # Late in document
        else:
            position_hints.append(0.5)  # Middle
        
        # Hint 3: Entry index within chunk (spread entries evenly)
        total_entries = chunk_info.get('total_entries', 10)  # Estimate if not available
        if total_entries > 0:
            index_ratio = entry_index / total_entries
            position_hints.append(index_ratio)
        
        # Calculate weighted average position
        if position_hints:
            avg_hint = sum(position_hints) / len(position_hints)
            estimated_position = chunk_start + int(chunk_length * avg_hint)
        else:
            # Final fallback: use entry index spacing
            estimated_position = chunk_start + (entry_index * 100)
        
        return estimated_position, 0.3  # Low confidence but better than random
    
    def _extract_search_candidates(self, entry: Dict) -> List[str]:
        """
        Extract multiple search candidates from entry for better matching
        """
        
        candidates = []
        key = (entry.get('key') or '').strip()
        value = (entry.get('value') or '').strip()
        
        if key and value:
            # Strategy 1: Exact key-value combinations
            candidates.extend([
                f"{key}: {value}",
                f"{key}:{value}",
                f"{key} = {value}",
                f"{key}={value}",
                f"{key} - {value}",
                f"{key} {value}"
            ])
            
            # Strategy 2: Key with partial value
            if len(value) > 20:
                candidates.extend([
                    f"{key}: {value[:20]}",
                    f"{key} {value[:20]}"
                ])
        
        if key:
            # Strategy 3: Just the key
            candidates.append(key)
            if key.split():
                candidates.append(key.split()[0])  # First word of key
        
        if value:
            # Strategy 4: Just the value
            candidates.append(value[:30])  # First 30 chars
            if value.split():
                candidates.append(value.split()[0])  # First word of value
                
            # Strategy 5: Individual words from value (first 3 words)
            value_words = value.split()[:3]
            for word in value_words:
                if len(word) > 3:  # Skip short words
                    candidates.append(word)
        
        # Remove empty and duplicate candidates, maintain order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in seen and len(candidate) > 2:
                seen.add(candidate)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _log_statistics(self, total_entries: int) -> None:
        """
        Log positioning method statistics
        """
        
        print(f"📊 Enhanced Method 1 Position Detection Results:")
        for method, count in self.method_stats.items():
            percentage = (count / total_entries) * 100 if total_entries > 0 else 0
            method_name = method.replace('_', ' ').title()
            print(f"   {method_name}: {count}/{total_entries} ({percentage:.1f}%)")
        
        # Calculate average confidence (approximate)
        high_confidence_methods = ['multi_candidate', 'fuzzy_match']
        high_confidence_count = sum(self.method_stats[m] for m in high_confidence_methods)
        avg_confidence = (high_confidence_count / total_entries) * 0.8 + \
                        (self.method_stats['context_aware'] / total_entries) * 0.7 + \
                        (self.method_stats['intelligent_fallback'] / total_entries) * 0.3
        
        print(f"   Estimated Average Confidence: {avg_confidence:.2f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get positioning statistics for debugging
        """
        
        total = sum(self.method_stats.values())
        return {
            'method_usage': self.method_stats.copy(),
            'total_entries': total,
            'success_rate': {
                'high_confidence': (self.method_stats['multi_candidate'] + self.method_stats['fuzzy_match']) / total if total > 0 else 0,
                'medium_confidence': self.method_stats['context_aware'] / total if total > 0 else 0,
                'low_confidence': self.method_stats['intelligent_fallback'] / total if total > 0 else 0
            }
        }


def fix_entry_ordering_enhanced_method_1(entries: List[Dict], full_text: str, chunks: List[Dict]) -> List[Dict]:
    """
    Convenience function to apply Enhanced Method 1 position tracking
    
    Args:
        entries: List of extracted entries
        full_text: Original PDF text
        chunks: List of chunk information
        
    Returns:
        List of entries sorted by document position
    """
    
    tracker = PositionTracker()
    return tracker.fix_entry_ordering(entries, full_text, chunks)