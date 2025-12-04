"""
Utility functions for PDF to Excel pipeline
Handles JSON cleaning, validation, and edge cases
"""

import json
import re
from typing import Dict, List, Any, Tuple


def clean_and_parse_json(llm_output: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response with markdown/text wrapper
    
    Handles common LLM output issues:
    - Markdown code blocks (```json ... ```)
    - Conversational text before/after JSON
    - Extra whitespace
    
    Args:
        llm_output: Raw output from LLM
    
    Returns:
        Parsed JSON object
    
    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    # Remove markdown code blocks
    clean_str = re.sub(r"```json|```", "", llm_output).strip()
    
    # Find the actual JSON object/array
    # Look for both object {} and array [] starts
    obj_start = clean_str.find('{')
    arr_start = clean_str.find('[')
    
    # Determine which comes first (or if only one exists)
    if obj_start == -1 and arr_start == -1:
        raise ValueError("No JSON object or array found in LLM output")
    
    if obj_start == -1:
        start = arr_start
        end = clean_str.rfind(']') + 1
    elif arr_start == -1:
        start = obj_start
        end = clean_str.rfind('}') + 1
    else:
        # Both exist, use whichever comes first
        start = min(obj_start, arr_start)
        if start == obj_start:
            end = clean_str.rfind('}') + 1
        else:
            end = clean_str.rfind(']') + 1
    
    if start != -1 and end != 0:
        clean_str = clean_str[start:end]
    
    return json.loads(clean_str)


def validate_json_structure(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that JSON has expected structure
    
    Args:
        data: Parsed JSON data
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for expected structure
    if isinstance(data, dict):
        if 'entries' in data:
            if not isinstance(data['entries'], list):
                return False, "'entries' must be a list"
            if len(data['entries']) == 0:
                return False, "'entries' list is empty"
            return True, ""
        else:
            # Fallback: treat entire dict as single entry
            return True, "Warning: No 'entries' key found, treating as single entry"
    elif isinstance(data, list):
        # Fallback: treat list as entries
        if len(data) == 0:
            return False, "Empty list returned"
        return True, "Warning: List format detected instead of object with 'entries'"
    else:
        return False, f"Unexpected data type: {type(data)}"


def normalize_keys(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize column keys across all entries to handle synonym issues
    
    This is a safety net - the prompt should handle this, but this provides backup
    
    Args:
        entries: List of data entries
    
    Returns:
        Entries with normalized keys
    """
    if not entries:
        return entries
    
    # Collect all unique keys
    all_keys = set()
    for entry in entries:
        all_keys.update(entry.keys())
    
    # Create a mapping of similar keys (basic implementation)
    # In production, you might use fuzzy matching or LLM-based normalization
    key_mapping = {}
    
    # Simple rule: lowercase and remove spaces/underscores for comparison
    def normalize_for_comparison(key: str) -> str:
        return key.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    # Group similar keys
    key_groups = {}
    for key in all_keys:
        normalized = normalize_for_comparison(key)
        if normalized not in key_groups:
            key_groups[normalized] = []
        key_groups[normalized].append(key)
    
    # For each group with multiple keys, pick the first one as canonical
    for normalized, keys in key_groups.items():
        if len(keys) > 1:
            canonical = keys[0]  # Use first occurrence
            for key in keys[1:]:
                key_mapping[key] = canonical
    
    # Apply mapping to all entries
    normalized_entries = []
    for entry in entries:
        normalized_entry = {}
        for key, value in entry.items():
            new_key = key_mapping.get(key, key)
            normalized_entry[new_key] = value
        normalized_entries.append(normalized_entry)
    
    return normalized_entries


def extract_entries_and_notes(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract entries and global notes from parsed JSON
    
    Handles multiple input formats:
    - {"entries": [...], "global_notes": "..."}
    - {"entries": [...]}
    - [...]  (list of entries)
    - {...}  (single entry)
    
    Args:
        data: Parsed JSON data
    
    Returns:
        Tuple of (entries_list, global_notes)
    """
    if isinstance(data, dict):
        if 'entries' in data:
            entries = data['entries']
            global_notes = data.get('global_notes', None)
        else:
            # Treat entire dict as single entry
            entries = [data]
            global_notes = None
    elif isinstance(data, list):
        entries = data
        global_notes = None
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    # Convert None to empty string for consistency
    if global_notes is None:
        global_notes = ""
    
    return entries, global_notes


def format_for_display(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format JSON for display in UI
    
    Args:
        data: JSON data
        indent: Indentation spaces
    
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)
