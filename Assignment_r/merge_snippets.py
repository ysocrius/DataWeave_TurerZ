"""
Merge multiple markdown snippet files into a single Python file
"""

import os
import re
from pathlib import Path

def extract_code_from_markdown(md_content: str) -> str:
    """Extract Python code from markdown code blocks"""
    # Find all code blocks
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, md_content, re.DOTALL)
    
    if matches:
        return matches[0]
    return ""

def merge_snippets(snippets_dir: str, output_file: str):
    """Merge all snippet files in order into a single Python file"""
    snippets_path = Path(snippets_dir)
    
    # Get all .md files sorted by name
    snippet_files = sorted(snippets_path.glob('*.md'))
    
    if not snippet_files:
        print(f"❌ No snippet files found in {snippets_dir}")
        return
    
    print(f"📝 Found {len(snippet_files)} snippet files:")
    for f in snippet_files:
        print(f"   - {f.name}")
    
    # Merge all snippets
    merged_code = []
    
    for snippet_file in snippet_files:
        print(f"\n🔄 Processing {snippet_file.name}...")
        
        with open(snippet_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        code = extract_code_from_markdown(md_content)
        
        if code:
            merged_code.append(code)
            print(f"   ✅ Extracted {len(code)} characters")
        else:
            print(f"   ⚠️  No code found in {snippet_file.name}")
    
    # Write merged code to output file
    output_path = Path(output_file)
    
    final_content = '\n\n'.join(merged_code)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"\n✅ Successfully merged {len(snippet_files)} snippets into {output_file}")
    print(f"📊 Total size: {len(final_content)} characters")

if __name__ == "__main__":
    # Configuration
    SNIPPETS_DIR = "Assignment_r/snippets"
    OUTPUT_FILE = "Assignment_r/realtime_pipeline.py"
    
    print("🚀 Starting snippet merge process...\n")
    merge_snippets(SNIPPETS_DIR, OUTPUT_FILE)
    print("\n✨ Merge complete!")
