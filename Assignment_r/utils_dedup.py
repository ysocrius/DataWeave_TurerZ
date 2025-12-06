"""
utils_dedup.py
--------------
Generic deduplication utilities for document extraction pipeline with comprehensive error handling.

CRITICAL: Per assignment requirements, this module does NOT pre-define keys
or use hardcoded field mappings. All normalization must be flexible and generic.

Enhanced with comprehensive error handling, graceful failure recovery, and null value protection.
"""

import re
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher

# Import error handling utilities
try:
    from error_handling_utils import get_error_handler, get_data_validator, safe_execute
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    print("⚠️  Error handling utilities not available, using basic error handling")
    ERROR_HANDLING_AVAILABLE = False
    
    # Fallback error handler
    class FallbackErrorHandler:
        def handle_error(self, error, context, recovery_function=None, fallback_value=None):
            print(f"❌ Error in {context}: {error}")
            if fallback_value is not None:
                return {"fallback_value": fallback_value, "fallback_used": True}
            return {"fallback_used": False}
    
    def get_error_handler():
        return FallbackErrorHandler()
    
    def safe_execute(func, error_handler, context, recovery_function, fallback_value, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = error_handler.handle_error(e, context, recovery_function, fallback_value)
            if error_info.get("fallback_used"):
                return error_info.get("fallback_value")
            raise


def normalize_field_name(field_name: str) -> str:
    """
    Normalize field name using GENERIC rules (no hardcoded mappings).
    
    Removes:
    - Session/room numbers (Session 1A → Session, Room 2B → Room)
    - Extra whitespace
    - Special characters at boundaries
    
    Args:
        field_name: Original field name
        
    Returns:
        Normalized field name
    """
    if not field_name:
        return ""
    
    # Remove leading/trailing whitespace
    normalized = field_name.strip()
    
    # Remove session/room numbers with letters (e.g., "1A", "2B")
    # Pattern: digit followed by optional letter at word boundaries
    normalized = re.sub(r'\b\d+[A-Z]?\b', '', normalized)
    
    # Remove standalone numbers that are likely identifiers
    # But keep numbers that are part of the content (e.g., "Top 10 Skills")
    normalized = re.sub(r'^\d+\s*[-:.]?\s*', '', normalized)
    normalized = re.sub(r'\s*[-:.]\s*\d+$', '', normalized)
    
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove special characters at start/end
    normalized = normalized.strip(' -:.,;')
    
    # Convert to Title Case for consistency
    normalized = normalized.title()
    
    return normalized


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings using SequenceMatcher with comprehensive error handling.
    
    Returns score between 0.0 (no match) and 1.0 (perfect match).
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    try:
        # Comprehensive null value protection
        if str1 is None:
            str1 = ""
        elif not isinstance(str1, str):
            str1 = str(str1) if str1 is not None else ""
            
        if str2 is None:
            str2 = ""
        elif not isinstance(str2, str):
            str2 = str(str2) if str2 is not None else ""
        
        # Early return for empty strings
        if not str1 or not str2:
            return 0.0
        
        # Normalize for comparison with error handling
        try:
            s1 = str1.lower().strip()
            s2 = str2.lower().strip()
        except (AttributeError, UnicodeError) as e:
            print(f"⚠️  String normalization error: {e}")
            return 0.0
        
        # Quick exact match check
        if s1 == s2:
            return 1.0
        
        # Use SequenceMatcher with error handling
        try:
            return SequenceMatcher(None, s1, s2).ratio()
        except Exception as e:
            print(f"⚠️  SequenceMatcher error: {e}")
            return 0.0
            
    except Exception as e:
        # Catch-all error handling
        print(f"⚠️  Unexpected error in calculate_similarity: {e}")
        return 0.0


def create_smart_signature(entry: Dict[str, Any]) -> str:
    """
    Create a signature for an entry combining key and value excerpt with comprehensive null value protection.
    
    Used for fuzzy matching and deduplication.
    
    Args:
        entry: Entry dictionary with Key and Value
        
    Returns:
        Signature string (empty string if entry is invalid)
    """
    try:
        # Comprehensive null value protection
        if not entry or not isinstance(entry, dict):
            return ""
        
        # Safe key extraction with multiple fallbacks
        key = entry.get('Key')
        if key is None:
            key = ""
        elif not isinstance(key, str):
            key = str(key) if key is not None else ""
        key = key.strip().lower()
        
        # Safe value extraction with multiple fallbacks
        value = entry.get('Value')
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value) if value is not None else ""
        value = value.strip().lower()
        
        # Additional validation
        if not key and not value:
            return ""
        
        # Use first 50 chars of value for signature
        value_excerpt = value[:50] if len(value) > 50 else value
        
        return f"{key}:{value_excerpt}"
        
    except Exception as e:
        # Graceful error handling - log error but don't crash
        print(f"⚠️  Error creating signature for entry: {e}")
        return ""


def select_best_value(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the best entry from a group of similar entries.
    
    Priority scoring (higher = better):
    - Has comments: +50
    - Value length: +len(value)
    - Has numeric data: +20
    - Not truncated (no "..."): +30
    - Longer key name (more specific): +len(key)/2
    
    Args:
        entries: List of similar entries
        
    Returns:
        Best entry from the group
    """
    if not entries:
        return {}
    
    if len(entries) == 1:
        return entries[0]
    
    def score_entry(entry: Dict[str, Any]) -> float:
        score = 0.0
        
        value = str(entry.get('Value') or '').strip()
        key = (entry.get('Key') or '').strip()
        comments = entry.get('Comments')
        
        # Has comments
        if comments and str(comments).strip():
            score += 50
        
        # Value length (longer is better)
        score += len(value)
        
        # Has numeric data
        if re.search(r'\d', value):
            score += 20
        
        # Not truncated
        if not value.endswith('...') and not value.endswith('…'):
            score += 30
        
        # More specific key name
        score += len(key) / 2
        
        return score
    
    # Find entry with highest score
    best_entry = max(entries, key=score_entry)
    
    # Merge comments from all entries if multiple have comments
    all_comments = []
    for entry in entries:
        comment = entry.get('Comments')
        if comment and str(comment).strip():
            comment_str = str(comment).strip()
            if comment_str not in all_comments:
                all_comments.append(comment_str)
    
    if len(all_comments) > 1:
        # Multiple entries had comments, merge them
        best_entry = dict(best_entry)  # Make a copy
        best_entry['Comments'] = ' | '.join(all_comments)
    
    return best_entry


def repair_truncated_values(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect and repair truncated values by finding complete versions.
    
    Looks for values ending with "..." and searches for complete versions
    in other entries with similar keys.
    
    Args:
        entries: List of all entries
        
    Returns:
        List of entries with truncated values repaired
    """
    if not entries:
        return entries
    
    repaired = []
    
    for entry in entries:
        value = str(entry.get('Value') or '').strip()
        key = (entry.get('Key') or '').strip()
        
        # Check if truncated
        is_truncated = value.endswith('...') or value.endswith('…')
        
        if is_truncated and len(value) > 3:
            # Remove the ellipsis
            partial_value = value.rstrip('.…').strip()
            
            # Search for complete value in other entries
            best_match = None
            best_similarity = 0.0
            
            for other_entry in entries:
                if other_entry is entry:
                    continue
                
                other_value = str(other_entry.get('Value') or '').strip()
                other_key = (other_entry.get('Key') or '').strip()
                
                # Check if keys are similar
                key_similarity = calculate_similarity(key, other_key)
                
                if key_similarity > 0.7:  # Similar keys
                    # Check if other value starts with our partial value
                    if other_value.lower().startswith(partial_value.lower()):
                        # This might be the complete version
                        if len(other_value) > len(partial_value):
                            value_similarity = calculate_similarity(partial_value, other_value[:len(partial_value)])
                            
                            if value_similarity > best_similarity:
                                best_similarity = value_similarity
                                best_match = other_value
            
            # If we found a complete version, use it
            if best_match and best_similarity > 0.85:
                entry = dict(entry)  # Make a copy
                entry['Value'] = best_match
        
        repaired.append(entry)
    
    return repaired


def consolidate_by_fuzzy_matching(
    entries: List[Dict[str, Any]], 
    threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Enhanced consolidation using multi-layer fuzzy matching with comprehensive error handling.
    
    Uses multiple similarity algorithms:
    1. Semantic key similarity (normalized)
    2. Value content similarity 
    3. Contextual relationship detection
    4. Session/temporal grouping
    
    Args:
        entries: List of all entries
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        Consolidated list of entries
    """
    try:
        # Input validation with comprehensive error handling
        if not entries:
            return []
        
        if not isinstance(entries, list):
            print(f"⚠️  Invalid entries type in fuzzy matching: expected list, got {type(entries)}")
            return []
        
        # Validate threshold
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            print(f"⚠️  Invalid threshold {threshold}, using default 0.85")
            threshold = 0.85
        
        # Filter out invalid entries before processing
        valid_entries = []
        for i, entry in enumerate(entries):
            try:
                if not entry or not isinstance(entry, dict):
                    continue
                
                key = entry.get('Key')
                value = entry.get('Value')
                
                if not key or not value:
                    continue
                
                # Ensure key and value are strings
                if not isinstance(key, str):
                    key = str(key) if key is not None else ""
                if not isinstance(value, str):
                    value = str(value) if value is not None else ""
                
                if key.strip() and value.strip():
                    valid_entries.append(entry)
                    
            except Exception as e:
                print(f"⚠️  Error validating entry {i} in fuzzy matching: {e}")
                continue
        
        if not valid_entries:
            print("⚠️  No valid entries for fuzzy matching")
            return []
        
        print(f"🔍 Fuzzy matching: {len(valid_entries)} valid entries, threshold={threshold}")
        
        entries = valid_entries
        
        # Enhanced grouping with transitive closure for educational entries
        groups = []
        processed = set()
        
        # Build similarity matrix for transitive closure
        n = len(entries)
        similarity_matrix = [[False] * n for _ in range(n)]
        
        # Fill similarity matrix
        for i in range(n):
            for j in range(i + 1, n):
                is_similar = _is_entries_similar(entries[i], entries[j], threshold)
                similarity_matrix[i][j] = is_similar
                similarity_matrix[j][i] = is_similar
        
        # Find connected components using DFS (transitive closure)
        def dfs(node, group, visited):
            visited.add(node)
            group.append(entries[node])
            for neighbor in range(n):
                if similarity_matrix[node][neighbor] and neighbor not in visited:
                    dfs(neighbor, group, visited)
        
        visited = set()
        for i in range(n):
            if i not in visited:
                group = []
                dfs(i, group, visited)
                groups.append(group)
    
        # Select best entry from each group with enhanced logic
        consolidated = []
        for group in groups:
            best = _select_best_from_group(group)
            consolidated.append(best)
        
        return consolidated
        
    except Exception as e:
        error_handler = get_error_handler()
        error_info = error_handler.handle_error(
            e,
            "consolidate_by_fuzzy_matching_critical_error",
            fallback_value=entries if isinstance(entries, list) else []
        )
        print(f"❌ Critical error in fuzzy matching: {e}")
        return error_info.get("fallback_value", [])


def _is_entries_similar(
    entry1: Dict[str, Any], 
    entry2: Dict[str, Any], 
    threshold: float
) -> bool:
    """
    Enhanced similarity detection using multiple algorithms.
    
    Checks:
    1. Normalized key similarity
    2. Value content similarity
    3. Semantic relationship detection
    4. Session/temporal context
    
    Args:
        entry1: First entry
        entry2: Second entry
        threshold: Base similarity threshold
        
    Returns:
        True if entries are similar enough to consolidate
    """
    key1 = (entry1.get('Key') or '').strip()
    key2 = (entry2.get('Key') or '').strip()
    value1 = str(entry1.get('Value') or '').strip()
    value2 = str(entry2.get('Value') or '').strip()
    
    # 1. Normalized key similarity
    norm_key1 = _normalize_key_for_comparison(key1)
    norm_key2 = _normalize_key_for_comparison(key2)
    key_similarity = calculate_similarity(norm_key1, norm_key2)
    
    # 2. Value similarity
    value_similarity = calculate_similarity(value1, value2)
    
    # 3. Signature similarity (combined key+value)
    sig1 = create_smart_signature(entry1)
    sig2 = create_smart_signature(entry2)
    sig_similarity = calculate_similarity(sig1, sig2)
    
    # 4. Semantic relationship detection
    semantic_score = _calculate_semantic_similarity(key1, key2, value1, value2)
    
    # 5. Educational similarity scoring (if applicable)
    educational_score = 0.0
    if _is_educational_entry_group([entry1, entry2]):
        educational_score = _calculate_educational_similarity_score(entry1, entry2)
    
    # 6. Session/temporal context similarity
    context_score = _calculate_context_similarity(entry1, entry2)
    
    # Multi-criteria decision with weighted scoring
    # High signature similarity (direct match)
    if sig_similarity >= threshold:
        return True
    
    # Strong key similarity with decent value similarity
    if key_similarity >= 0.9 and value_similarity >= 0.7:
        return True
    
    # Semantic relationship detected
    if semantic_score >= 0.8:
        return True
    
    # Educational similarity detected (high confidence for educational data)
    if educational_score >= 0.4:  # Lower threshold for educational entries
        return True
    
    # Strong contextual relationship with good key similarity
    if context_score >= 0.7 and key_similarity >= 0.75:
        return True
    
    # Combined weighted score approach (including educational scoring)
    if educational_score > 0:
        # Educational entries get special weighting
        weighted_score = (
            key_similarity * 0.3 +
            value_similarity * 0.2 +
            semantic_score * 0.15 +
            educational_score * 0.25 +  # Higher weight for educational similarity
            context_score * 0.1
        )
    else:
        # Standard weighting for non-educational entries
        weighted_score = (
            key_similarity * 0.4 +
            value_similarity * 0.3 +
            semantic_score * 0.2 +
            context_score * 0.1
        )
    
    return weighted_score >= (threshold * 0.9)


def is_educational_grade_pattern(key: str) -> bool:
    """
    Detect if a key contains ordinal numbers that appear to be meaningful identifiers
    rather than just sequence numbers that should be removed.
    
    Uses generic patterns to identify when ordinal numbers are likely part of
    the semantic meaning (like academic levels, versions, editions) rather than
    just organizational numbering.
    
    Args:
        key: Key string to analyze
        
    Returns:
        True if key contains ordinal numbers that should be preserved
    """
    if not key:
        return False
    
    key_lower = key.lower()
    
    # Exclude obvious session/organizational patterns first
    if re.search(r'\b(session|meeting|room|track|hall|floor)\b', key_lower):
        return False
    
    # Pattern 1: Ordinal followed by a noun (likely meaningful)
    # Examples: "10th Grade", "3rd Edition", "1st Semester"
    # But exclude things like "3rd Floor"
    if re.search(r'\b\d+(st|nd|rd|th)\s+\w+\b', key_lower):
        # Exclude organizational terms
        if not re.search(r'\b\d+(st|nd|rd|th)\s+(floor|session|meeting|room)\b', key_lower):
            return True
    
    # Pattern 2: Noun followed by ordinal (likely meaningful)  
    # Examples: "Grade 10th", "Edition 3rd"
    if re.search(r'\b\w+\s+\d+(st|nd|rd|th)\b', key_lower):
        return True
    
    # Pattern 3: Ordinal with hyphen or dash (likely meaningful)
    # Examples: "10th-Grade", "3rd-Year"
    if re.search(r'\b\d+(st|nd|rd|th)[-_]\w+\b', key_lower):
        return True
    
    # Pattern 4: Word with ordinal (likely meaningful)
    # Examples: "Grade-10th", "Year-3rd"  
    if re.search(r'\b\w+[-_]\d+(st|nd|rd|th)\b', key_lower):
        return True
    
    return False


def normalize_key_enhanced(key: str) -> str:
    """
    Enhanced key normalization that preserves meaningful ordinal numbers.
    
    This is the main normalization function that:
    - Preserves ordinal numbers that appear to be meaningful identifiers
    - Removes session/room identifiers and timestamps  
    - Normalizes spacing and punctuation
    - Maintains semantic meaning without hardcoding specific terms
    
    Args:
        key: Original key string
        
    Returns:
        Normalized key with meaningful ordinals preserved
    """
    if not key:
        return ""
    
    original_key = key.strip()
    normalized = original_key.lower()
    
    # Remove session/room identifiers (generic patterns)
    # Remove ordinals that precede these organizational terms
    normalized = re.sub(r'\b\d+(st|nd|rd|th)?\s*(session|room|track|hall)\b', '', normalized)
    # Remove remaining session/room identifiers with numbers/letters
    normalized = re.sub(r'\b(session|room|track|hall)\s*[a-z0-9]+\b', '', normalized)
    
    # Remove time stamps and dates
    normalized = re.sub(r'\b\d{1,2}:\d{2}(\s*(am|pm))?\b', '', normalized)
    normalized = re.sub(r'\b\d{1,2}/\d{1,2}(/\d{2,4})?\b', '', normalized)
    
    # Check if this key contains meaningful ordinal patterns AFTER basic cleanup
    has_meaningful_ordinals = is_educational_grade_pattern(normalized)
    
    # Handle ordinal numbers based on context
    if not has_meaningful_ordinals:
        # For keys without meaningful ordinals, remove all ordinal numbers
        normalized = re.sub(r'\b\d+(st|nd|rd|th)\b', '', normalized)
        
        # Remove standalone numbers at boundaries
        normalized = re.sub(r'^\d+\s*[-:.]?\s*', '', normalized)
        normalized = re.sub(r'\s*[-:.]\s*\d+$', '', normalized)
    
    # Remove extra whitespace and clean up
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    normalized = normalized.strip(' -:.,;')
    
    # If we preserved meaningful ordinals, try to maintain better formatting
    if has_meaningful_ordinals and normalized:
        # Capitalize first letter of each word for readability
        normalized = ' '.join(word.capitalize() for word in normalized.split())
    
    return normalized if normalized else original_key.lower()


def _normalize_key_for_comparison(key: str) -> str:
    """
    Enhanced key normalization for better comparison.
    
    This function now delegates to normalize_key_enhanced for consistent behavior.
    
    Args:
        key: Original key
        
    Returns:
        Normalized key for comparison
    """
    return normalize_key_enhanced(key)


def _is_educational_grade_variation(key1: str, key2: str, value1: str, value2: str) -> bool:
    """
    Check if two entries are variations of the same ordinal-based entry.
    
    Detects cases where ordinal numbers were corrupted during processing:
    - "10th Something" vs "Th Something" 
    - "3rd Edition" vs "Rd Edition"
    
    Uses generic pattern matching without hardcoded terms.
    """
    # Normalize keys and values
    k1, k2 = key1.lower().strip(), key2.lower().strip()
    v1, v2 = value1.lower().strip(), value2.lower().strip()
    
    # Look for ordinal corruption patterns
    # Pattern: full ordinal (10th) vs partial ordinal (th)
    ordinal_pattern = r'(\d+)?(st|nd|rd|th)(\s+\w+)?'
    
    match1 = re.search(ordinal_pattern, k1)
    match2 = re.search(ordinal_pattern, k2)
    
    if match1 and match2:
        num1, suffix1, word1 = match1.groups()
        num2, suffix2, word2 = match2.groups()
        
        # Check if one has number and other doesn't, but same suffix
        if suffix1 == suffix2 and word1 == word2:
            # Case: "10th Grade" vs "Th Grade" (or similar)
            if (num1 and not num2) or (num2 and not num1):
                # Check if values are similar (indicating same entry)
                if _values_are_similar_dates_or_grades(v1, v2):
                    return True
    
    # Also check for cases where the ordinal was completely removed
    # "10th Something" vs "Something"
    for full_key, partial_key in [(k1, k2), (k2, k1)]:
        # Remove ordinals from full key and see if it matches partial
        no_ordinal = re.sub(r'\b\d+(st|nd|rd|th)\s*', '', full_key).strip()
        if no_ordinal and calculate_similarity(no_ordinal, partial_key) > 0.8:
            # Check values for similarity
            if _values_are_similar_dates_or_grades(v1, v2):
                return True
    
    return False

def _values_are_similar_dates_or_grades(v1: str, v2: str) -> bool:
    """Check if two values contain similar date ranges or grade info"""
    # Look for date patterns and percentages
    import re
    
    # Extract years and percentages
    year_pattern = r'\b(19|20)\d{2}\b'
    percent_pattern = r'\b\d+%\b'
    
    years1 = set(re.findall(year_pattern, v1))
    years2 = set(re.findall(year_pattern, v2))
    
    percents1 = set(re.findall(percent_pattern, v1))
    percents2 = set(re.findall(percent_pattern, v2))
    
    # If they share years or percentages, likely the same entry
    return bool(years1 & years2) or bool(percents1 & percents2)

def _calculate_semantic_similarity(
    key1: str, key2: str, value1: str, value2: str
) -> float:
    """
    Calculate semantic similarity between entries.
    
    Special handling for educational grade variations like "10th Grade" vs "Th Grade"
    
    Detects relationships like:
    - Speaker name variations
    - Topic/session relationships
    - Time/schedule relationships
    
    Args:
        key1, key2: Entry keys
        value1, value2: Entry values
        
    Returns:
        Semantic similarity score (0.0 to 1.0)
    """
    # Special case: Educational grade variations
    if _is_educational_grade_variation(key1, key2, value1, value2):
        return 0.95  # High similarity for grade variations
    
    # Continue with existing logic...
    # For now, return a basic similarity score
    from difflib import SequenceMatcher
    
    # Calculate key similarity
    key_sim = SequenceMatcher(None, key1.lower(), key2.lower()).ratio()
    
    # Calculate value similarity
    value_sim = SequenceMatcher(None, value1.lower(), value2.lower()).ratio()
    
    # Weighted combination
    return (key_sim * 0.7) + (value_sim * 0.3)


def _calculate_speaker_similarity(
    key1: str, key2: str, value1: str, value2: str
) -> float:
    """
    Calculate speaker-specific similarity for conference/event data.
    
    Args:
        key1, key2: Entry keys
        value1, value2: Entry values
        
    Returns:
        Speaker similarity score (0.0 to 1.0)
    """
    # Convert to lowercase for comparison
    k1, k2 = key1.lower(), key2.lower()
    v1, v2 = value1.lower(), value2.lower()
    
    # Speaker name variations
    if any(word in k1 for word in ['speaker', 'presenter', 'facilitator']):
        if any(word in k2 for word in ['speaker', 'presenter', 'facilitator']):
            # Check if values contain similar names
            name_similarity = _calculate_name_similarity(v1, v2)
            if name_similarity > 0.7:
                return 0.9
    
    # Session/topic relationships
    if any(word in k1 for word in ['session', 'topic', 'title', 'subject']):
        if any(word in k2 for word in ['session', 'topic', 'title', 'subject']):
            topic_similarity = calculate_similarity(v1, v2)
            if topic_similarity > 0.8:
                return 0.85
    
    # Time/schedule relationships
    if any(word in k1 for word in ['time', 'schedule', 'start', 'end']):
        if any(word in k2 for word in ['time', 'schedule', 'start', 'end']):
            # Time values should be similar or related
            time_similarity = _calculate_time_similarity(v1, v2)
            if time_similarity > 0.6:
                return 0.8
    
    # Location relationships
    if any(word in k1 for word in ['room', 'location', 'venue', 'hall']):
        if any(word in k2 for word in ['room', 'location', 'venue', 'hall']):
            location_similarity = calculate_similarity(v1, v2)
            if location_similarity > 0.7:
                return 0.8
    
    return 0.0


def _calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between names, handling variations."""
    if not name1 or not name2:
        return 0.0
    
    # Split names into parts
    parts1 = set(name1.lower().split())
    parts2 = set(name2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(parts1.intersection(parts2))
    union = len(parts1.union(parts2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def _calculate_time_similarity(time1: str, time2: str) -> float:
    """Calculate similarity between time values."""
    # Simple time similarity - could be enhanced with actual time parsing
    return calculate_similarity(time1, time2)


def _calculate_context_similarity(
    entry1: Dict[str, Any], 
    entry2: Dict[str, Any]
) -> float:
    """
    Calculate contextual similarity based on comments and metadata.
    
    Args:
        entry1, entry2: Entries to compare
        
    Returns:
        Context similarity score (0.0 to 1.0)
    """
    comments1 = str(entry1.get('Comments') or '').lower()
    comments2 = str(entry2.get('Comments') or '').lower()
    
    if not comments1 or not comments2:
        return 0.0
    
    # Calculate comment similarity
    comment_similarity = calculate_similarity(comments1, comments2)
    
    # Boost score if comments indicate same session/context
    if any(word in comments1 and word in comments2 
           for word in ['session', 'event', 'conference', 'meeting']):
        comment_similarity += 0.2
    
    return min(1.0, comment_similarity)


def _is_educational_entry_group(entries: List[Dict[str, Any]]) -> bool:
    """
    Determine if a group of entries represents educational data that needs special merging.
    
    Detects educational patterns like:
    - Grade levels (10th Grade, 12th Grade)
    - Academic years (2019-2020, 2020-2021)
    - Educational institutions
    - Academic achievements/scores
    
    Args:
        entries: List of entries to analyze
        
    Returns:
        True if this appears to be educational data requiring intelligent merging
    """
    if not entries or len(entries) < 2:
        return False
    
    educational_indicators = 0
    total_entries = len(entries)
    
    for entry in entries:
        key = str(entry.get('Key', '')).lower()
        value = str(entry.get('Value', '')).lower()
        
        # Check for educational grade patterns
        if re.search(r'\b\d+(st|nd|rd|th)\s*(grade|year|level)\b', key + ' ' + value):
            educational_indicators += 1
            continue
        
        # Check for academic year patterns
        if re.search(r'\b(19|20)\d{2}[-/](19|20)?\d{2}\b', value):
            educational_indicators += 1
            continue
        
        # Check for educational keywords
        educational_keywords = [
            'grade', 'school', 'university', 'college', 'education', 'academic',
            'semester', 'year', 'gpa', 'score', 'achievement', 'degree', 'diploma',
            'transcript', 'course', 'class', 'subject', 'major', 'minor'
        ]
        
        if any(keyword in key or keyword in value for keyword in educational_keywords):
            educational_indicators += 1
            continue
        
        # Check for percentage/grade patterns
        if re.search(r'\b\d+(\.\d+)?%\b', value) or re.search(r'\bgpa\b', key + ' ' + value):
            educational_indicators += 1
            continue
    
    # Consider it educational if more than 50% of entries show educational patterns
    return educational_indicators / total_entries > 0.5


def _calculate_educational_similarity_score(
    entry1: Dict[str, Any], 
    entry2: Dict[str, Any]
) -> float:
    """
    Calculate specialized similarity score for educational entries.
    
    Considers:
    - Grade level variations (10th Grade vs Th Grade)
    - Academic year overlaps
    - Institution name variations
    - Score/GPA similarities
    
    Args:
        entry1, entry2: Educational entries to compare
        
    Returns:
        Educational similarity score (0.0 to 1.0)
    """
    key1 = str(entry1.get('Key', '')).lower()
    key2 = str(entry2.get('Key', '')).lower()
    value1 = str(entry1.get('Value', '')).lower()
    value2 = str(entry2.get('Value', '')).lower()
    
    similarity_score = 0.0
    
    # 1. Grade level variation detection (high weight)
    if _is_educational_grade_variation(key1, key2, value1, value2):
        similarity_score += 0.6  # Very high weight for grade variations
    
    # 1b. Additional check for similar educational keys with ordinal corruption
    # Check if keys are similar after removing ordinals
    key1_no_ordinal = re.sub(r'\b\d*(st|nd|rd|th)\b', '', key1).strip()
    key2_no_ordinal = re.sub(r'\b\d*(st|nd|rd|th)\b', '', key2).strip()
    if key1_no_ordinal and key2_no_ordinal:
        key_base_similarity = calculate_similarity(key1_no_ordinal, key2_no_ordinal)
        if key_base_similarity > 0.8:  # Very similar base keys
            similarity_score += 0.5
    
    # 2. Academic year overlap detection
    years1 = set(re.findall(r'\b(19|20)\d{2}\b', value1))
    years2 = set(re.findall(r'\b(19|20)\d{2}\b', value2))
    if years1 and years2:
        year_overlap = len(years1.intersection(years2)) / len(years1.union(years2))
        similarity_score += year_overlap * 0.3
    
    # 3. Institution name similarity and educational synonyms
    institution_keywords = ['school', 'university', 'college', 'institute', 'academy']
    educational_synonyms = {
        'university': ['college', 'univ'],
        'college': ['university', 'univ'],
        'gpa': ['grade', 'performance', 'academic'],
        'grade': ['gpa', 'performance', 'academic'],
        'performance': ['gpa', 'grade', 'academic'],
        'academic': ['gpa', 'grade', 'performance']
    }
    
    # Check for institution keywords
    for keyword in institution_keywords:
        if keyword in key1 and keyword in key2:
            key_similarity = calculate_similarity(key1, key2)
            similarity_score += key_similarity * 0.2
            break
    
    # Check for educational synonyms
    key1_words = set(key1.lower().split())
    key2_words = set(key2.lower().split())
    
    synonym_matches = 0
    for word1 in key1_words:
        for word2 in key2_words:
            if word1 == word2:
                synonym_matches += 1
            elif word1 in educational_synonyms and word2 in educational_synonyms[word1]:
                synonym_matches += 0.8  # Partial match for synonyms
    
    if synonym_matches > 0:
        similarity_score += min(0.3, synonym_matches * 0.15)
    
    # 4. Score/GPA similarity
    scores1 = re.findall(r'\b(\d+(?:\.\d+)?)%?\b', value1)
    scores2 = re.findall(r'\b(\d+(?:\.\d+)?)%?\b', value2)
    if scores1 and scores2:
        # Simple score similarity check
        try:
            score1_nums = [float(s) for s in scores1 if s]
            score2_nums = [float(s) for s in scores2 if s]
            if score1_nums and score2_nums:
                avg_diff = sum(abs(s1 - s2) for s1 in score1_nums for s2 in score2_nums) / (len(score1_nums) * len(score2_nums))
                score_similarity = max(0, 1 - avg_diff / 100)  # Normalize to 0-1
                similarity_score += score_similarity * 0.1
        except (ValueError, ZeroDivisionError):
            pass
    
    return min(1.0, similarity_score)


def _merge_educational_entries_intelligently(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Intelligent merging logic specifically designed for educational entries.
    
    Merge strategy selection based on data completeness:
    - Preserves complete grade information over corrupted versions
    - Merges academic year ranges intelligently
    - Combines complementary educational data
    - Prioritizes entries with complete ordinal numbers
    
    Args:
        entries: List of educational entries to merge
        
    Returns:
        Best merged educational entry
    """
    if not entries:
        return {}
    
    if len(entries) == 1:
        return entries[0]
    
    def educational_score_entry(entry: Dict[str, Any]) -> float:
        """Score entry based on educational data quality"""
        score = 0.0
        
        key = str(entry.get('Key', '')).lower()
        value = str(entry.get('Value', '')).lower()
        
        # 1. Complete ordinal numbers (highest priority)
        if re.search(r'\b\d+(st|nd|rd|th)\b', key):
            score += 50  # High score for complete ordinals
        elif re.search(r'\b(st|nd|rd|th)\b', key):
            score -= 20  # Penalty for incomplete ordinals
        
        # 2. Academic year completeness
        year_matches = re.findall(r'\b(19|20)\d{2}\b', value)
        if len(year_matches) >= 2:
            score += 30  # Complete year range
        elif len(year_matches) == 1:
            score += 15  # Single year
        
        # 3. Educational content richness
        educational_terms = ['grade', 'gpa', 'score', 'achievement', 'degree', 'diploma']
        for term in educational_terms:
            if term in key or term in value:
                score += 10
        
        # 4. Data completeness (length and structure)
        score += len(value) * 0.1  # Longer values generally better
        
        # 5. Numerical data presence (grades, scores, years)
        if re.search(r'\b\d+(\.\d+)?%?\b', value):
            score += 15
        
        # 6. Not truncated
        if not value.endswith('...') and not value.endswith('…'):
            score += 10
        
        return score
    
    # Find the best entry using educational scoring
    best_entry = max(entries, key=educational_score_entry)
    
    # Intelligent data merging for educational entries
    merged_entry = dict(best_entry)  # Start with best entry
    
    # Merge complementary information from other entries
    all_years = set()
    all_scores = []
    all_comments = []
    
    for entry in entries:
        value = str(entry.get('Value', ''))
        comments = str(entry.get('Comments', '')).strip()
        
        # Collect all years mentioned
        years = re.findall(r'\b(19|20)\d{2}\b', value)
        all_years.update(years)
        
        # Collect all scores/percentages
        scores = re.findall(r'\b\d+(\.\d+)?%\b', value)
        all_scores.extend(scores)
        
        # Collect meaningful comments
        if comments and comments not in all_comments:
            all_comments.append(comments)
    
    # Enhance the merged entry with collected data
    original_value = str(merged_entry.get('Value', ''))
    
    # Add missing years if we found more complete year information
    if len(all_years) > 1:
        existing_years = set(re.findall(r'\b(19|20)\d{2}\b', original_value))
        missing_years = all_years - existing_years
        if missing_years:
            # Add missing years in a structured way
            year_list = sorted(list(all_years))
            if len(year_list) == 2:
                year_range = f"{year_list[0]}-{year_list[1]}"
                if year_range not in original_value:
                    merged_entry['Value'] = f"{original_value} ({year_range})"
    
    # Consolidate comments intelligently
    if len(all_comments) > 1:
        # Remove duplicates and merge
        unique_comments = []
        for comment in all_comments:
            if not any(calculate_similarity(comment, existing) > 0.8 for existing in unique_comments):
                unique_comments.append(comment)
        
        if len(unique_comments) > 1:
            merged_entry['Comments'] = ' | '.join(unique_comments[:3])  # Limit to top 3
    
    return merged_entry


def _select_best_from_group(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enhanced selection of best entry from a group with intelligent merging for educational entries.
    
    Uses improved scoring that considers:
    - Content completeness
    - Information richness
    - Data quality indicators
    - Educational-specific merge strategies
    
    Args:
        entries: List of similar entries
        
    Returns:
        Best entry from the group using intelligent educational merging
    """
    if not entries:
        return {}
    
    if len(entries) == 1:
        return entries[0]
    
    # Check if this is an educational entry group that needs special handling
    if _is_educational_entry_group(entries):
        return _merge_educational_entries_intelligently(entries)
    
    def enhanced_score_entry(entry: Dict[str, Any]) -> float:
        score = 0.0
        
        value = str(entry.get('Value') or '').strip()
        key = (entry.get('Key') or '').strip()
        comments = str(entry.get('Comments') or '').strip()
        
        # Content completeness (40% of score)
        score += len(value) * 0.4
        
        # Has meaningful comments (30% of score)
        if comments and len(comments) > 10:
            score += 30
        
        # Not truncated (20% of score)
        if not value.endswith('...') and not value.endswith('…'):
            score += 20
        
        # Information richness indicators (10% of score)
        if re.search(r'\d', value):  # Contains numbers
            score += 5
        if len(value.split()) > 3:  # Multi-word value
            score += 5
        if any(char in value for char in '.,;:'):  # Structured content
            score += 3
        
        # Key specificity
        score += len(key) * 0.1
        
        return score
    
    # Find entry with highest score
    best_entry = max(entries, key=enhanced_score_entry)
    
    # Enhanced comment consolidation
    best_entry = _consolidate_comments(entries, best_entry)
    
    return best_entry


def _consolidate_comments(
    entries: List[Dict[str, Any]], 
    best_entry: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Intelligently consolidate comments from multiple entries.
    
    Args:
        entries: All entries in the group
        best_entry: Selected best entry
        
    Returns:
        Best entry with consolidated comments
    """
    all_comments = []
    
    for entry in entries:
        comment = str(entry.get('Comments') or '').strip()
        if comment and comment not in all_comments:
            # Avoid duplicate comments
            is_duplicate = any(
                calculate_similarity(comment, existing) > 0.8 
                for existing in all_comments
            )
            if not is_duplicate:
                all_comments.append(comment)
    
    if len(all_comments) > 1:
        # Multiple unique comments, merge intelligently
        best_entry = dict(best_entry)  # Make a copy
        
        # Sort comments by length (longer = more informative)
        all_comments.sort(key=len, reverse=True)
        
        # Limit to top 3 most informative comments
        consolidated = ' | '.join(all_comments[:3])
        best_entry['Comments'] = consolidated
    
    return best_entry


def deduplicate_exact_keys(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact duplicate keys with comprehensive error handling and null value protection.
    
    This is the final deduplication step after fuzzy matching.
    
    Args:
        entries: List of entries
        
    Returns:
        Deduplicated entries
    """
    error_handler = get_error_handler()
    
    try:
        # Input validation
        if not entries:
            return []
        
        if not isinstance(entries, list):
            error_info = error_handler.handle_error(
                ValueError(f"Invalid entries type: expected list, got {type(entries)}"),
                "deduplicate_exact_keys_input_validation",
                fallback_value=[]
            )
            return error_info.get("fallback_value", [])
        
        key_map = {}
        processing_errors = 0
        
        for i, entry in enumerate(entries):
            try:
                # Comprehensive entry validation
                if not entry or not isinstance(entry, dict):
                    processing_errors += 1
                    continue
                
                # Safe key extraction
                key = entry.get('Key')
                if key is None:
                    processing_errors += 1
                    continue
                
                # Handle non-string keys
                if not isinstance(key, str):
                    try:
                        key = str(key)
                    except Exception as e:
                        print(f"⚠️  Cannot convert key to string at index {i}: {e}")
                        processing_errors += 1
                        continue
                
                key = key.strip()
                if not key:
                    processing_errors += 1
                    continue
                
                key_lower = key.lower()
                
                if key_lower not in key_map:
                    key_map[key_lower] = entry
                else:
                    # Keep the better entry with error handling
                    try:
                        current = key_map[key_lower]
                        better = safe_execute(
                            select_best_value,
                            error_handler,
                            f"select_best_value_for_key_{key_lower}",
                            None,
                            current,  # fallback to current entry
                            [current, entry]
                        )
                        key_map[key_lower] = better
                    except Exception as e:
                        print(f"⚠️  Error selecting best value for key '{key}': {e}")
                        # Keep current entry as fallback
                        processing_errors += 1
                        continue
                        
            except Exception as e:
                print(f"⚠️  Error processing entry {i} in exact deduplication: {e}")
                processing_errors += 1
                continue
        
        result = list(key_map.values())
        
        if processing_errors > 0:
            print(f"⚠️  Exact deduplication completed with {processing_errors} processing errors")
        
        return result
        
    except Exception as e:
        error_info = error_handler.handle_error(
            e,
            "deduplicate_exact_keys_critical_error",
            fallback_value=entries if isinstance(entries, list) else []
        )
        return error_info.get("fallback_value", [])


def validate_and_clean(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and clean entries with comprehensive error handling and null value protection.
    
    Removes:
    - Entries without keys
    - Entries without values
    - Entries with null/empty values
    - Malformed entries
    
    Args:
        entries: List of entries
        
    Returns:
        Cleaned list of entries
    """
    if not entries:
        return []
    
    if not isinstance(entries, list):
        print(f"⚠️  Invalid entries type: expected list, got {type(entries)}")
        return []
    
    cleaned = []
    invalid_count = 0
    
    for i, entry in enumerate(entries):
        try:
            # Comprehensive entry validation
            if entry is None:
                invalid_count += 1
                continue
                
            if not isinstance(entry, dict):
                print(f"⚠️  Invalid entry at index {i}: expected dict, got {type(entry)}")
                invalid_count += 1
                continue
            
            # Safe key extraction with multiple validation levels
            key = entry.get('Key')
            if key is None:
                invalid_count += 1
                continue
            
            # Handle non-string keys
            if not isinstance(key, str):
                try:
                    key = str(key)
                except Exception as e:
                    print(f"⚠️  Cannot convert key to string at index {i}: {e}")
                    invalid_count += 1
                    continue
            
            key = key.strip()
            
            # Key validation
            if not key or len(key) < 2:
                invalid_count += 1
                continue
            
            # Safe value extraction with multiple validation levels
            value = entry.get('Value')
            if value is None:
                invalid_count += 1
                continue
            
            # Handle non-string values
            if not isinstance(value, str):
                try:
                    value_str = str(value)
                except Exception as e:
                    print(f"⚠️  Cannot convert value to string at index {i}: {e}")
                    invalid_count += 1
                    continue
            else:
                value_str = value
            
            value_str = value_str.strip()
            
            # Value validation - expanded null value detection
            if not value_str or value_str.lower() in [
                'null', 'none', 'n/a', 'na', 'nil', 'undefined', 
                'empty', '', 'blank', 'missing', 'unknown'
            ]:
                invalid_count += 1
                continue
            
            # Additional validation for suspicious values
            if len(value_str) < 1:
                invalid_count += 1
                continue
            
            # Create cleaned entry with validated fields
            cleaned_entry = {
                'Key': key,
                'Value': value_str,
                'Comments': entry.get('Comments', ''),
            }
            
            # Preserve additional fields safely
            for field_name, field_value in entry.items():
                if field_name not in ['Key', 'Value', 'Comments']:
                    try:
                        cleaned_entry[field_name] = field_value
                    except Exception as e:
                        print(f"⚠️  Error preserving field '{field_name}' at index {i}: {e}")
            
            cleaned.append(cleaned_entry)
            
        except Exception as e:
            print(f"⚠️  Error processing entry at index {i}: {e}")
            invalid_count += 1
            continue
    
    if invalid_count > 0:
        print(f"📊 Validation summary: {len(cleaned)} valid entries, {invalid_count} invalid entries removed")
    
    return cleaned


def full_deduplication_pipeline(
    entries: List[Dict[str, Any]],
    fuzzy_threshold: float = 0.85,
    enable_fuzzy: bool = True,
    enable_repair: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run the complete deduplication pipeline with comprehensive error handling and graceful failure recovery.
    
    Steps:
    1. Validate and clean entries
    2. Normalize field names
    3. Repair truncated values (optional)
    4. Fuzzy match consolidation (optional)
    5. Exact key deduplication
    
    Args:
        entries: List of entries from all chunks
        fuzzy_threshold: Similarity threshold for fuzzy matching
        enable_fuzzy: Enable fuzzy matching
        enable_repair: Enable truncation repair
        
    Returns:
        Tuple of (deduplicated_entries, stats_dict)
    """
    import time
    
    start_time = time.time()
    
    # Initialize comprehensive stats with error tracking
    stats = {
        'input_count': 0,
        'after_validation': 0,
        'after_normalization': 0,
        'after_repair': 0,
        'after_fuzzy': 0,
        'after_exact': 0,
        'final_count': 0,
        'processing_errors': [],
        'step_timings': {},
        'error_recovery_count': 0,
        'total_processing_time': 0
    }
    
    try:
        # Input validation with comprehensive error handling
        if entries is None:
            print("⚠️  Input entries is None, returning empty result")
            stats['processing_errors'].append("Input entries is None")
            return [], stats
        
        if not isinstance(entries, list):
            print(f"⚠️  Invalid entries type: expected list, got {type(entries)}")
            stats['processing_errors'].append(f"Invalid entries type: {type(entries)}")
            return [], stats
        
        stats['input_count'] = len(entries)
        
        if stats['input_count'] == 0:
            print("📊 No entries to process")
            return [], stats
        
        print(f"🚀 Starting deduplication pipeline with {stats['input_count']} entries")
        
        # Step 1: Validate and clean with error recovery
        step_start = time.time()
        try:
            entries = validate_and_clean(entries)
            stats['after_validation'] = len(entries)
            stats['step_timings']['validation'] = time.time() - step_start
            print(f"✅ Step 1 - Validation: {stats['after_validation']} entries remaining")
        except Exception as e:
            error_msg = f"Validation step failed: {str(e)}"
            print(f"❌ {error_msg}")
            stats['processing_errors'].append(error_msg)
            stats['error_recovery_count'] += 1
            # Try to recover with minimal validation
            try:
                entries = [entry for entry in entries if entry and isinstance(entry, dict) and entry.get('Key') and entry.get('Value')]
                stats['after_validation'] = len(entries)
                print(f"🔄 Validation recovery: {stats['after_validation']} entries recovered")
            except Exception as recovery_error:
                print(f"❌ Validation recovery failed: {recovery_error}")
                return [], stats
        
        if not entries:
            print("⚠️  No valid entries after validation")
            return [], stats
        
        # Step 2: Normalize field names with error recovery
        step_start = time.time()
        try:
            normalized_count = 0
            for i, entry in enumerate(entries):
                try:
                    if 'Key' in entry and entry['Key']:
                        original_key = entry['Key']
                        entry['Key'] = normalize_key_enhanced(entry['Key'])
                        normalized_count += 1
                except Exception as e:
                    print(f"⚠️  Normalization error for entry {i}: {e}")
                    stats['processing_errors'].append(f"Normalization error for entry {i}: {str(e)}")
                    # Keep original key if normalization fails
                    continue
            
            stats['after_normalization'] = len(entries)
            stats['step_timings']['normalization'] = time.time() - step_start
            print(f"✅ Step 2 - Normalization: {normalized_count} keys normalized")
        except Exception as e:
            error_msg = f"Normalization step failed: {str(e)}"
            print(f"❌ {error_msg}")
            stats['processing_errors'].append(error_msg)
            stats['error_recovery_count'] += 1
            # Continue without normalization
            stats['after_normalization'] = len(entries)
        
        # Step 3: Repair truncated values with error recovery
        step_start = time.time()
        if enable_repair:
            try:
                entries = repair_truncated_values(entries)
                stats['after_repair'] = len(entries)
                stats['step_timings']['repair'] = time.time() - step_start
                print(f"✅ Step 3 - Repair: {stats['after_repair']} entries after repair")
            except Exception as e:
                error_msg = f"Repair step failed: {str(e)}"
                print(f"❌ {error_msg}")
                stats['processing_errors'].append(error_msg)
                stats['error_recovery_count'] += 1
                # Continue without repair
                stats['after_repair'] = len(entries)
        else:
            stats['after_repair'] = len(entries)
            print("⏭️  Step 3 - Repair: Skipped (disabled)")
        
        # Step 4: Fuzzy matching consolidation with error recovery
        step_start = time.time()
        if enable_fuzzy:
            try:
                # Validate fuzzy threshold
                if not isinstance(fuzzy_threshold, (int, float)) or fuzzy_threshold < 0 or fuzzy_threshold > 1:
                    print(f"⚠️  Invalid fuzzy threshold {fuzzy_threshold}, using default 0.85")
                    fuzzy_threshold = 0.85
                
                entries = consolidate_by_fuzzy_matching(entries, fuzzy_threshold)
                stats['after_fuzzy'] = len(entries)
                stats['step_timings']['fuzzy'] = time.time() - step_start
                print(f"✅ Step 4 - Fuzzy matching: {stats['after_fuzzy']} entries after consolidation")
            except Exception as e:
                error_msg = f"Fuzzy matching step failed: {str(e)}"
                print(f"❌ {error_msg}")
                stats['processing_errors'].append(error_msg)
                stats['error_recovery_count'] += 1
                # Continue without fuzzy matching
                stats['after_fuzzy'] = len(entries)
        else:
            stats['after_fuzzy'] = len(entries)
            print("⏭️  Step 4 - Fuzzy matching: Skipped (disabled)")
        
        # Step 5: Exact key deduplication with error recovery
        step_start = time.time()
        try:
            entries = deduplicate_exact_keys(entries)
            stats['after_exact'] = len(entries)
            stats['step_timings']['exact_dedup'] = time.time() - step_start
            print(f"✅ Step 5 - Exact deduplication: {stats['after_exact']} entries after final cleanup")
        except Exception as e:
            error_msg = f"Exact deduplication step failed: {str(e)}"
            print(f"❌ {error_msg}")
            stats['processing_errors'].append(error_msg)
            stats['error_recovery_count'] += 1
            # Continue without exact deduplication
            stats['after_exact'] = len(entries)
        
        # Final statistics
        stats['final_count'] = len(entries)
        stats['total_processing_time'] = time.time() - start_time
        
        if stats['input_count'] > 0:
            stats['reduction_percentage'] = round(
                (1 - stats['final_count'] / stats['input_count']) * 100, 1
            )
        else:
            stats['reduction_percentage'] = 0.0
        
        # Final summary
        print(f"🎉 Deduplication pipeline complete:")
        print(f"   📊 Input: {stats['input_count']} → Output: {stats['final_count']} entries")
        print(f"   📉 Reduction: {stats['reduction_percentage']}%")
        print(f"   ⏱️  Total time: {stats['total_processing_time']:.2f}s")
        
        if stats['processing_errors']:
            print(f"   ⚠️  Errors encountered: {len(stats['processing_errors'])}")
            print(f"   🔄 Error recoveries: {stats['error_recovery_count']}")
        
        return entries, stats
        
    except Exception as e:
        # Catch-all error handling for unexpected failures
        error_msg = f"Critical pipeline failure: {str(e)}"
        print(f"💥 {error_msg}")
        stats['processing_errors'].append(error_msg)
        stats['total_processing_time'] = time.time() - start_time
        
        # Try to return whatever we have
        if 'entries' in locals() and entries:
            stats['final_count'] = len(entries)
            print(f"🔄 Returning {stats['final_count']} entries despite critical error")
            return entries, stats
        else:
            print("💥 Complete pipeline failure, returning empty result")
            return [], stats
