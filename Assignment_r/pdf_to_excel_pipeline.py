"""
pdf_to_excel_pipeline.py
------------------------
Main pipeline orchestrator for PDF to Excel conversion
Handles: PDF extraction -> LLM processing -> Excel generation
"""

import pdfplumber
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# Import custom modules
from prompt_templates import get_extraction_prompt, get_expected_format_prompt
from utils import clean_and_parse_json, extract_entries_and_notes

# Load environment variables
load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF with layout-aware parameters
    
    Handles multi-column layouts and preserves structure
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text as string
    """
    full_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Adjust tolerance for multi-column layouts
            # x_tolerance: horizontal spacing between characters
            # y_tolerance: vertical spacing between lines
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            
            if text:
                full_text.append(f"--- Page {page_num} ---\n{text}")
    
    return "\n\n".join(full_text)


def process_with_llm(text_content: str) -> str:
    """
    Send extracted text to LLM for structured extraction
    
    Args:
        text_content: Extracted PDF text
    
    Returns:
        Raw LLM response (JSON string)
    """
    # Import OpenAI client directly (more reliable than LangChain)
    from openai import OpenAI
    
    # Get API key and model from environment
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Build prompt (use expected format prompt for better matching)
    prompt = get_expected_format_prompt(text_content)
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    
    return response.choices[0].message.content


def convert_to_excel(llm_output: str, output_path: str) -> tuple[pd.DataFrame, str]:
    """
    Convert LLM JSON response to Excel with global notes handling
    
    CRITICAL: Single sheet only (automated grading requirement)
    
    Args:
        llm_output: Raw LLM response
        output_path: Path to save Excel file
    
    Returns:
        Tuple of (DataFrame, global_notes)
    """
    # Clean and parse JSON
    parsed_data = clean_and_parse_json(llm_output)
    
    # Extract entries and global notes
    entries, global_notes = extract_entries_and_notes(parsed_data)
    
    # Create DataFrame
    df = pd.DataFrame(entries)
    
    # CRITICAL: Add global notes to first row's Comments column (single sheet only)
    if global_notes and global_notes.strip():
        if 'Comments' not in df.columns:
            df['Comments'] = None
        
        # Prepend global note to first row's comments
        first_comment = df.loc[0, 'Comments']
        if pd.isna(first_comment) or not first_comment:
            df.loc[0, 'Comments'] = f"[GLOBAL NOTE: {global_notes}]"
        else:
            df.loc[0, 'Comments'] = f"[GLOBAL NOTE: {global_notes}] | {first_comment}"
    
    # Save to Excel (single sheet)
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    return df, global_notes


def main(input_pdf: str, output_excel: str = "Output.xlsx") -> None:
    """
    Main pipeline execution
    
    Args:
        input_pdf: Path to input PDF file
        output_excel: Path to output Excel file
    """
    print(f"🔄 Processing: {input_pdf}")
    
    # Step 1: Extract text
    print("📖 Extracting text from PDF...")
    text_content = extract_text_from_pdf(input_pdf)
    print(f"✅ Extracted {len(text_content)} characters")
    
    # Step 2: Process with LLM
    print("🤖 Sending to AI for analysis...")
    llm_response = process_with_llm(text_content)
    print("✅ AI processing complete")
    
    # Step 3: Convert to Excel
    print("📊 Converting to Excel format...")
    df, global_notes = convert_to_excel(llm_response, output_excel)
    print(f"✅ Generated Excel with {len(df)} rows and {len(df.columns)} columns")
    
    if global_notes:
        print(f"📝 Global Note: {global_notes}")
    
    print(f"💾 Saved to: {output_excel}")
    print("\n✨ Processing complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_excel_pipeline.py <input_pdf> [output_excel]")
        print("Example: python pdf_to_excel_pipeline.py 'Data Input.pdf' 'Output.xlsx'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "Output.xlsx"
    
    main(input_file, output_file)
