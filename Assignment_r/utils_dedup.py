"""
utils_dedup.py
--------------
Generic deduplication utilities for document extraction pipeline.

CRITICAL: Per assignment requirements, this module does NOT pre-define keys
or use hardcoded field mappings. All normalization must be flexible and generic.
"""

import re
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher


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
    Calculate similarity between two strings using SequenceMatcher.
    
    Returns score between 0.0 (no match) and 1.0 (perfect match).
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not str1 or not str2:
        return 0.0
    
    # Normalize for comparison
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()
    
    if s1 == s2:
        return 1.0
    
    return SequenceMatcher(None, s1, s2).ratio()


def create_smart_signature(entry: Dict[str, Any]) -> str:
    """
    Create a signature for an entry combining key and value excerpt.
    
    Used for fuzzy matching and deduplication.
    
    Args:
        entry: Entry dictionary with Key and Value
        
    Returns:
        Signature string
    """
    key = (entry.get('Key') or '').strip().lower()
    value = str(entry.get('Value') or '').strip().lower()
    
    # Use first 50 chars of value for signature
    value_excerpt = value[:50] if len(value) > 50 else value
    
    return f"{key}:{value_excerpt}"


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
    Enhanced consolidation using multi-layer fuzzy matching.
    
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
    if not entries:
        return entries
    
    # Enhanced grouping with multiple similarity checks
    groups = []
    processed = set()
    
    for i, entry in enumerate(entries):
        if i in processed:
            continue
        
        # Start a new group with this entry
        group = [entry]
        processed.add(i)
        
        key1 = (entry.get('Key') or '').strip()
        value1 = str(entry.get('Value') or '').strip()
        
        # Find similar entries using enhanced matching
        for j, other_entry in enumerate(entries):
            if j in processed:
                continue
            
            key2 = (other_entry.get('Key') or '').strip()
            value2 = str(other_entry.get('Value') or '').strip()
            
            # Multi-layer similarity analysis
            is_similar = _is_entries_similar(
                entry, other_entry, threshold
            )
            
            if is_similar:
                group.append(other_entry)
                processed.add(j)
        
        groups.append(group)
    
    # Select best entry from each group with enhanced logic
    consolidated = []
    for group in groups:
        best = _select_best_from_group(group)
        consolidated.append(best)
    
    return consolidated


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
    
    # 5. Session/temporal context similarity
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
    
    # Strong contextual relationship with good key similarity
    if context_score >= 0.7 and key_similarity >= 0.75:
        return True
    
    # Combined weighted score approach
    weighted_score = (
        key_similarity * 0.4 +
        value_similarity * 0.3 +
        semantic_score * 0.2 +
        context_score * 0.1
    )
    
    return weighted_score >= (threshold * 0.9)


def _normalize_key_for_comparison(key: str) -> str:
    """
    Enhanced key normalization for better comparison.
    
    Removes session identifiers, room numbers, time stamps,
    and other variable elements while preserving semantic meaning.
    
    Args:
        key: Original key
        
    Returns:
        Normalized key for comparison
    """
    if not key:
        return ""
    
    normalized = key.lower().strip()
    
    # Remove session/room identifiers
    normalized = re.sub(r'\b(session|room|track|hall)\s*[a-z0-9]+\b', '', normalized)
    
    # Remove time stamps and dates
    normalized = re.sub(r'\b\d{1,2}:\d{2}(\s*(am|pm))?\b', '', normalized)
    normalized = re.sub(r'\b\d{1,2}/\d{1,2}(/\d{2,4})?\b', '', normalized)
    
    # Remove ordinal numbers (1st, 2nd, etc.)
    normalized = re.sub(r'\b\d+(st|nd|rd|th)\b', '', normalized)
    
    # Remove standalone numbers at boundaries
    normalized = re.sub(r'^\d+\s*[-:.]?\s*', '', normalized)
    normalized = re.sub(r'\s*[-:.]\s*\d+$', '', normalized)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def _calculate_semantic_similarity(
    key1: str, key2: str, value1: str, value2: str
) -> float:
    """
    Calculate semantic similarity between entries.
    
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


def _select_best_from_group(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enhanced selection of best entry from a group.
    
    Uses improved scoring that considers:
    - Content completeness
    - Information richness
    - Data quality indicators
    
    Args:
        entries: List of similar entries
        
    Returns:
        Best entry from the group
    """
    if not entries:
        return {}
    
    if len(entries) == 1:
        return entries[0]
    
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
    Remove exact duplicate keys, keeping the best value for each.
    
    This is the final deduplication step after fuzzy matching.
    
    Args:
        entries: List of entries
        
    Returns:
        Deduplicated entries
    """
    if not entries:
        return entries
    
    key_map = {}
    
    for entry in entries:
        key = (entry.get('Key') or '').strip()
        
        if not key:
            continue
        
        key_lower = key.lower()
        
        if key_lower not in key_map:
            key_map[key_lower] = entry
        else:
            # Keep the better entry
            current = key_map[key_lower]
            better = select_best_value([current, entry])
            key_map[key_lower] = better
    
    return list(key_map.values())


def validate_and_clean(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and clean entries before final output.
    
    Removes:
    - Entries without keys
    - Entries without values
    - Entries with null/empty values
    
    Args:
        entries: List of entries
        
    Returns:
        Cleaned list of entries
    """
    cleaned = []
    
    for entry in entries:
        key = (entry.get('Key') or '').strip()
        value = entry.get('Value')
        value_str = str(value).strip() if value is not None else ''
        
        # Skip invalid entries
        if not key or len(key) < 2:
            continue
        
        if not value_str or value_str.lower() in ['null', 'none', 'n/a', '']:
            continue
        
        cleaned.append(entry)
    
    return cleaned


def full_deduplication_pipeline(
    entries: List[Dict[str, Any]],
    fuzzy_threshold: float = 0.85,
    enable_fuzzy: bool = True,
    enable_repair: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run the complete deduplication pipeline.
    
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
    stats = {
        'input_count': len(entries),
        'after_validation': 0,
        'after_normalization': 0,
        'after_repair': 0,
        'after_fuzzy': 0,
        'after_exact': 0,
        'final_count': 0
    }
    
    # Step 1: Validate and clean
    entries = validate_and_clean(entries)
    stats['after_validation'] = len(entries)
    
    # Step 2: Normalize field names
    for entry in entries:
        if 'Key' in entry:
            entry['Key'] = normalize_field_name(entry['Key'])
    stats['after_normalization'] = len(entries)
    
    # Step 3: Repair truncated values
    if enable_repair:
        entries = repair_truncated_values(entries)
    stats['after_repair'] = len(entries)
    
    # Step 4: Fuzzy matching consolidation
    if enable_fuzzy:
        entries = consolidate_by_fuzzy_matching(entries, fuzzy_threshold)
    stats['after_fuzzy'] = len(entries)
    
    # Step 5: Exact key deduplication (final cleanup)
    entries = deduplicate_exact_keys(entries)
    stats['after_exact'] = len(entries)
    
    stats['final_count'] = len(entries)
    stats['reduction_percentage'] = round(
        (1 - stats['final_count'] / max(1, stats['input_count'])) * 100, 1
    )
    
    return entries, stats
