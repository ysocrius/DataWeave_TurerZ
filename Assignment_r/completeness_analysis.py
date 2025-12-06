#!/usr/bin/env python3
"""
PDF Completeness Analysis Tool
Analyzes what percentage of the original PDF content is actually captured in the JSON
and what types of information are lost in the extraction process.
"""

import json
import sys
import os
from PyPDF2 import PdfReader
import re
from typing import Dict, List, Any, Set
from collections import Counter

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
    """Normalize text for analysis"""
    # Remove extra whitespace, normalize line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()

def extract_words(text: str) -> Set[str]:
    """Extract unique words from text"""
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return set(words)

def analyze_content_types(pdf_text: str) -> Dict[str, Any]:
    """Analyze different types of content in the PDF"""
    analysis = {
        "total_characters": len(pdf_text),
        "total_words": len(pdf_text.split()),
        "unique_words": len(extract_words(pdf_text)),
        "content_types": {
            "dates": len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+ \d{4}\b', pdf_text)),
            "percentages": len(re.findall(r'\d+\.?\d*%', pdf_text)),
            "phone_numbers": len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pdf_text)),
            "emails": len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pdf_text)),
            "numbers": len(re.findall(r'\b\d+\b', pdf_text)),
            "urls": len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', pdf_text)),
        },
        "sections": {
            "paragraphs": len([p for p in pdf_text.split('\n\n') if len(p.strip()) > 50]),
            "bullet_points": len(re.findall(r'^\s*[•\-\*]\s+', pdf_text, re.MULTILINE)),
            "headings": len(re.findall(r'^[A-Z][A-Z\s]{5,}$', pdf_text, re.MULTILINE)),
        }
    }
    return analysis

def calculate_coverage(pdf_text: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate how much of the PDF content is covered by the JSON extraction"""
    
    # Get all text from JSON entries
    json_text = ""
    entries = json_data.get('entries', [])
    
    for entry in entries:
        json_text += f"{entry.get('Key', '')} {entry.get('Value', '')} {entry.get('Comments', '')} "
    
    # Add global notes if present
    if json_data.get('global_notes'):
        json_text += json_data['global_notes']
    
    # Normalize both texts
    pdf_normalized = normalize_text(pdf_text)
    json_normalized = normalize_text(json_text)
    
    # Word-level analysis
    pdf_words = extract_words(pdf_text)
    json_words = extract_words(json_text)
    
    word_coverage = len(json_words.intersection(pdf_words)) / len(pdf_words) if pdf_words else 0
    
    # Character-level analysis
    char_coverage = len(json_normalized) / len(pdf_normalized) if pdf_normalized else 0
    
    # Find what's missing
    missing_words = pdf_words - json_words
    
    return {
        "word_coverage": word_coverage,
        "character_coverage": char_coverage,
        "total_pdf_words": len(pdf_words),
        "total_json_words": len(json_words),
        "captured_words": len(json_words.intersection(pdf_words)),
        "missing_words": len(missing_words),
        "missing_word_examples": list(missing_words)[:20],  # First 20 missing words
    }

def identify_lost_information(pdf_text: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Identify what types of information are lost in extraction"""
    
    pdf_analysis = analyze_content_types(pdf_text)
    
    # Count what was captured in JSON
    json_text = ""
    entries = json_data.get('entries', [])
    for entry in entries:
        json_text += f"{entry.get('Key', '')} {entry.get('Value', '')} {entry.get('Comments', '')} "
    
    json_analysis = analyze_content_types(json_text)
    
    lost_info = {
        "formatting": {
            "description": "Visual formatting, layout, fonts, colors",
            "impact": "High - Document structure and emphasis lost"
        },
        "content_gaps": {},
        "structural_elements": {
            "tables": "Cannot detect table structures from text extraction",
            "images": "All images and graphics lost",
            "charts": "Charts and diagrams not captured",
            "headers_footers": "Page headers and footers may be lost"
        }
    }
    
    # Compare content types
    for content_type, pdf_count in pdf_analysis["content_types"].items():
        json_count = json_analysis["content_types"].get(content_type, 0)
        if pdf_count > 0:
            capture_rate = json_count / pdf_count
            lost_info["content_gaps"][content_type] = {
                "pdf_count": pdf_count,
                "json_count": json_count,
                "capture_rate": capture_rate,
                "lost_count": pdf_count - json_count
            }
    
    return lost_info

def analyze_completeness(pdf_path: str, json_path: str) -> Dict[str, Any]:
    """Main completeness analysis function"""
    print(f"🔍 Analyzing completeness between:")
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
    
    # Perform analyses
    pdf_analysis = analyze_content_types(pdf_text)
    coverage_analysis = calculate_coverage(pdf_text, json_data)
    lost_info = identify_lost_information(pdf_text, json_data)
    
    return {
        "pdf_analysis": pdf_analysis,
        "coverage_analysis": coverage_analysis,
        "lost_information": lost_info,
        "entries_count": len(json_data.get('entries', [])),
        "processing_time": json_data.get('processing_time', 'N/A')
    }

def print_completeness_report(results: Dict[str, Any]):
    """Print detailed completeness report"""
    print("=" * 80)
    print("📊 PDF COMPLETENESS ANALYSIS REPORT")
    print("=" * 80)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    pdf_analysis = results["pdf_analysis"]
    coverage = results["coverage_analysis"]
    lost_info = results["lost_information"]
    
    print(f"📄 Original PDF Analysis:")
    print(f"   Total Characters: {pdf_analysis['total_characters']:,}")
    print(f"   Total Words: {pdf_analysis['total_words']:,}")
    print(f"   Unique Words: {pdf_analysis['unique_words']:,}")
    print(f"   Paragraphs: {pdf_analysis['sections']['paragraphs']}")
    print(f"   Bullet Points: {pdf_analysis['sections']['bullet_points']}")
    print(f"   Headings: {pdf_analysis['sections']['headings']}")
    print()
    
    print(f"📊 Extraction Coverage:")
    print(f"   Word Coverage: {coverage['word_coverage']:.1%}")
    print(f"   Character Coverage: {coverage['character_coverage']:.1%}")
    print(f"   Words Captured: {coverage['captured_words']:,} / {coverage['total_pdf_words']:,}")
    print(f"   Words Missing: {coverage['missing_words']:,}")
    print()
    
    print(f"❌ What's Lost in Extraction:")
    print(f"   📋 Formatting & Layout: Complete loss of visual structure")
    print(f"   🖼️  Images & Graphics: All visual elements lost")
    print(f"   📊 Tables & Charts: Structure and relationships lost")
    print(f"   📝 Document Flow: Narrative and reading order may be disrupted")
    print()
    
    print(f"🔍 Content Type Analysis:")
    for content_type, data in lost_info["content_gaps"].items():
        if data["pdf_count"] > 0:
            print(f"   {content_type.title()}: {data['json_count']}/{data['pdf_count']} captured ({data['capture_rate']:.1%})")
    print()
    
    print(f"💡 Key Insights:")
    word_cov = coverage['word_coverage']
    if word_cov < 0.3:
        print(f"   ⚠️  LOW COVERAGE ({word_cov:.1%}) - Most content not captured")
    elif word_cov < 0.6:
        print(f"   ⚠️  MODERATE COVERAGE ({word_cov:.1%}) - Significant content missing")
    else:
        print(f"   ✅ GOOD COVERAGE ({word_cov:.1%}) - Most content captured")
    
    print(f"   📝 The JSON contains structured data points, not the full document")
    print(f"   🔄 You CANNOT recreate the original PDF from this JSON")
    print(f"   🎯 This is DATA EXTRACTION, not document preservation")
    print()
    
    print(f"🎯 What This Means:")
    print(f"   ✅ Excellent for: Extracting specific data points and facts")
    print(f"   ❌ Poor for: Document reconstruction or preserving layout")
    print(f"   💡 Use case: Converting documents to structured databases")
    
    if coverage['missing_words'] > 0:
        print(f"\n🔍 Sample Missing Words:")
        missing_sample = coverage['missing_word_examples'][:10]
        print(f"   {', '.join(missing_sample)}")

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python completeness_analysis.py <pdf_path> <json_path>")
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
    
    # Run analysis
    results = analyze_completeness(pdf_path, json_path)
    print_completeness_report(results)

if __name__ == "__main__":
    main()