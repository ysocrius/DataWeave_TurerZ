#!/usr/bin/env python3
"""
PDF to JSON Fidelity Verification Tool
Compares extracted JSON data with original PDF content to verify extraction accuracy
"""

import json
import sys
import os
from pathlib import Path
from PyPDF2 import PdfReader
import re
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""

def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load extracted JSON data"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return {}

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Remove extra whitespace, normalize line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\/\@\#\$\%\&\*\+\=\<\>\?\!\'\"]', '', text)
    return text.lower()

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings"""
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()

def find_text_in_pdf(pdf_text: str, search_text: str, context_chars: int = 100) -> List[Tuple[str, float]]:
    """Find text in PDF with context and similarity score"""
    if not search_text.strip():
        return []
    
    normalized_pdf = normalize_text(pdf_text)
    normalized_search = normalize_text(search_text)
    
    # Direct substring search
    if normalized_search in normalized_pdf:
        start_idx = normalized_pdf.find(normalized_search)
        context_start = max(0, start_idx - context_chars)
        context_end = min(len(pdf_text), start_idx + len(search_text) + context_chars)
        context = pdf_text[context_start:context_end]
        return [(context, 1.0)]
    
    # Fuzzy matching for partial matches
    words = normalized_search.split()
    if len(words) > 1:
        matches = []
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 3 and phrase in normalized_pdf:
                    start_idx = normalized_pdf.find(phrase)
                    context_start = max(0, start_idx - context_chars)
                    context_end = min(len(pdf_text), start_idx + len(phrase) + context_chars)
                    context = pdf_text[context_start:context_end]
                    similarity = len(phrase) / len(normalized_search)
                    matches.append((context, similarity))
        
        # Return best matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:3]
    
    return []

def verify_fidelity(pdf_path: str, json_path: str) -> Dict[str, Any]:
    """Main fidelity verification function"""
    print(f"🔍 Verifying fidelity between:")
    print(f"   📄 PDF: {pdf_path}")
    print(f"   📊 JSON: {json_path}")
    print()
    
    # Load data
    pdf_text = extract_text_from_pdf(pdf_path)
    json_data = load_json_data(json_path)
    
    if not pdf_text:
        return {"error": "Could not extract text from PDF"}
    
    if not json_data:
        return {"error": "Could not load JSON data"}
    
    # Get entries from JSON
    entries = json_data.get('entries', [])
    if not entries:
        return {"error": "No entries found in JSON data"}
    
    print(f"📊 Analysis Summary:")
    print(f"   📄 PDF text length: {len(pdf_text):,} characters")
    print(f"   📊 JSON entries: {len(entries)}")
    print(f"   ⏱️  Processing time: {json_data.get('processing_time', 'N/A')}s")
    print(f"   🔧 Session ID: {json_data.get('session_id', 'N/A')}")
    print()
    
    # Verify each entry
    verification_results = {
        "total_entries": len(entries),
        "verified_entries": 0,
        "missing_entries": 0,
        "partial_matches": 0,
        "perfect_matches": 0,
        "entries_analysis": [],
        "overall_fidelity": 0.0,
        "pdf_coverage": 0.0
    }
    
    print("🔍 Entry-by-Entry Verification:")
    print("=" * 80)
    
    for i, entry in enumerate(entries, 1):
        key = entry.get('Key', '')
        value = entry.get('Value', '')
        comments = entry.get('Comments', '')
        
        print(f"\n📝 Entry {i}: {key}")
        print(f"   Value: {value}")
        if comments:
            print(f"   Comments: {comments}")
        
        # Search for key in PDF
        key_matches = find_text_in_pdf(pdf_text, key)
        value_matches = find_text_in_pdf(pdf_text, value)
        
        entry_analysis = {
            "entry_number": i,
            "key": key,
            "value": value,
            "key_found": len(key_matches) > 0,
            "value_found": len(value_matches) > 0,
            "key_similarity": key_matches[0][1] if key_matches else 0.0,
            "value_similarity": value_matches[0][1] if value_matches else 0.0,
            "overall_similarity": 0.0
        }
        
        # Calculate overall similarity for this entry
        if key_matches and value_matches:
            entry_analysis["overall_similarity"] = (entry_analysis["key_similarity"] + entry_analysis["value_similarity"]) / 2
            verification_results["perfect_matches"] += 1
            print(f"   ✅ PERFECT MATCH - Key: {entry_analysis['key_similarity']:.2f}, Value: {entry_analysis['value_similarity']:.2f}")
        elif key_matches or value_matches:
            entry_analysis["overall_similarity"] = max(entry_analysis["key_similarity"], entry_analysis["value_similarity"])
            verification_results["partial_matches"] += 1
            print(f"   ⚠️  PARTIAL MATCH - Key: {entry_analysis['key_similarity']:.2f}, Value: {entry_analysis['value_similarity']:.2f}")
        else:
            verification_results["missing_entries"] += 1
            print(f"   ❌ NOT FOUND - Neither key nor value found in PDF")
        
        # Show context for found items
        if key_matches:
            print(f"   🔍 Key context: ...{key_matches[0][0][:100]}...")
        if value_matches:
            print(f"   🔍 Value context: ...{value_matches[0][0][:100]}...")
        
        verification_results["entries_analysis"].append(entry_analysis)
    
    # Calculate overall metrics
    verification_results["verified_entries"] = verification_results["perfect_matches"] + verification_results["partial_matches"]
    
    if verification_results["total_entries"] > 0:
        verification_results["overall_fidelity"] = (
            verification_results["perfect_matches"] * 1.0 + 
            verification_results["partial_matches"] * 0.5
        ) / verification_results["total_entries"]
    
    # Calculate PDF coverage (how much of the PDF content is represented in the JSON)
    total_pdf_chars = len(normalize_text(pdf_text))
    covered_chars = 0
    
    for entry in entries:
        key_text = normalize_text(entry.get('Key', ''))
        value_text = normalize_text(entry.get('Value', ''))
        covered_chars += len(key_text) + len(value_text)
    
    if total_pdf_chars > 0:
        verification_results["pdf_coverage"] = min(1.0, covered_chars / total_pdf_chars)
    
    return verification_results

def print_final_report(results: Dict[str, Any]):
    """Print final verification report"""
    print("\n" + "=" * 80)
    print("📊 FINAL FIDELITY REPORT")
    print("=" * 80)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    total = results["total_entries"]
    perfect = results["perfect_matches"]
    partial = results["partial_matches"]
    missing = results["missing_entries"]
    fidelity = results["overall_fidelity"]
    coverage = results["pdf_coverage"]
    
    print(f"📊 Entry Statistics:")
    print(f"   Total Entries: {total}")
    print(f"   Perfect Matches: {perfect} ({perfect/total*100:.1f}%)")
    print(f"   Partial Matches: {partial} ({partial/total*100:.1f}%)")
    print(f"   Missing/Not Found: {missing} ({missing/total*100:.1f}%)")
    print()
    
    print(f"🎯 Fidelity Metrics:")
    print(f"   Overall Fidelity Score: {fidelity:.2%}")
    print(f"   PDF Coverage Estimate: {coverage:.2%}")
    print()
    
    # Fidelity assessment
    if fidelity >= 0.9:
        print("✅ EXCELLENT FIDELITY - Extraction is highly accurate")
    elif fidelity >= 0.75:
        print("✅ GOOD FIDELITY - Extraction is mostly accurate with minor issues")
    elif fidelity >= 0.5:
        print("⚠️  MODERATE FIDELITY - Extraction has some accuracy issues")
    else:
        print("❌ POOR FIDELITY - Extraction has significant accuracy issues")
    
    print()
    print("💡 Recommendations:")
    if missing > total * 0.2:
        print("   • Consider adjusting chunking strategy for better coverage")
    if partial > perfect:
        print("   • Review LLM prompts for more precise extraction")
    if coverage < 0.5:
        print("   • PDF may contain non-extractable content (images, tables)")
    
    print("\n🔍 For detailed analysis, review the entry-by-entry verification above.")

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python verify_fidelity.py <pdf_path> <json_path>")
        print("Example: python verify_fidelity.py document.pdf extracted_data.json")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    json_path = sys.argv[2]
    
    # Validate file paths
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        sys.exit(1)
    
    # Run verification
    results = verify_fidelity(pdf_path, json_path)
    print_final_report(results)

if __name__ == "__main__":
    main()