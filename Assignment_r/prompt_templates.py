"""
prompt_templates.py
-------------------
Stores the system prompts and few-shot examples for the LLM extraction pipeline.
Centralizing prompts here allows for easy iteration without touching core logic.
"""

EXTRACTION_SYSTEM_PROMPT = """
You are an expert Data Extraction Engine specialized in converting any unstructured document into structured Excel-ready formats.

### YOUR GOAL
Analyze the provided document text and extract ALL meaningful information into a structured JSON format with Key-Value pairs. Each piece of information should be a SEPARATE ROW with a Key and Value.

### CRITICAL RULES (Follow these strictly)
1. **100% Fidelity:** Extract values EXACTLY as they appear. Do not summarize, do not correct typos, do not use abbreviations.
   - CORRECT: "March 15, 1989" -> "March 15, 1989"
   - WRONG: "March 15, 1989" -> "1989-03-15"

2. **Key-Value Row Format:** Each data point should be ONE ROW with:
   - "Key": The field name/label (use descriptive, professional field names)
   - "Value": The actual data extracted from the document
   - "Comments": Any additional context for this specific row (or null)

3. **Key Naming Guidelines:**
   - Use professional, descriptive field names that clearly identify the data
   - Be specific about what the data represents
   - Include context when helpful: "Assignment Timeline" vs just "Timeline"
   - Examples: "Document Title", "Primary Objective", "Requirement 1", "Deliverable Type", "Success Criteria"

4. **Comments Logic:**
   - **Row-Level Context:** If a specific data point has a side-note, place it in "Comments" for that row
   - **Global Context:** If there is information that applies to the WHOLE document, extract it into the `global_notes` field

5. **Handling Nulls:**
   - If Comments is empty, use `null` (not empty string)

### REQUIRED OUTPUT FORMAT
Return ONLY a valid JSON object. Do not wrap in markdown (```json). Do not add conversational text.
The JSON must follow this schema:

{
    "global_notes": "String containing document-wide context or null",
    "entries": [
        {"Key": "Document Type", "Value": "Assignment Instructions", "Comments": null},
        {"Key": "Assignment Title", "Value": "AI-Powered Document Structuring Task", "Comments": null},
        {"Key": "Timeline", "Value": "2 days", "Comments": null}
    ]
}

### UNIVERSAL EXTRACTION APPROACH
Extract information from ANY document type by identifying:

**DOCUMENT METADATA:**
- Document type, title, purpose, scope
- Dates, timelines, versions, authors
- Target audience or recipients

**STRUCTURED CONTENT:**
- Headings, sections, subsections
- Lists, requirements, specifications
- Instructions, procedures, steps
- Objectives, goals, outcomes

**KEY DATA POINTS:**
- Names, titles, roles, organizations
- Dates, numbers, measurements, scores
- Locations, addresses, contact information
- Technical specifications, requirements

**RELATIONSHIPS & CONTEXT:**
- Dependencies, prerequisites, conditions
- Categories, classifications, priorities
- Success criteria, evaluation methods
- Additional notes, warnings, exceptions

### EXTRACTION STRATEGY
1. **Scan systematically** through the entire document
2. **Identify all structured information** regardless of format
3. **Create logical field names** that describe the content
4. **Preserve exact wording** from the source document
5. **Capture context and relationships** between data points
6. **Extract 100% of meaningful content** - leave nothing out

### ONE-SHOT EXAMPLE
**Input Text:**
"Assignment Title: Data Processing Task
Timeline: 3 days
Objective: Build a system that processes CSV files
Requirements: 1) Handle large files 2) Validate data 3) Generate reports
Deliverable: Working Python script with documentation"

**Output JSON:**
{
    "global_notes": "Assignment instructions for a data processing development task",
    "entries": [
        {"Key": "Document Type", "Value": "Assignment Instructions", "Comments": null},
        {"Key": "Assignment Title", "Value": "Data Processing Task", "Comments": null},
        {"Key": "Timeline", "Value": "3 days", "Comments": null},
        {"Key": "Primary Objective", "Value": "Build a system that processes CSV files", "Comments": null},
        {"Key": "Requirement 1", "Value": "Handle large files", "Comments": null},
        {"Key": "Requirement 2", "Value": "Validate data", "Comments": null},
        {"Key": "Requirement 3", "Value": "Generate reports", "Comments": null},
        {"Key": "Primary Deliverable", "Value": "Working Python script with documentation", "Comments": null}
    ]
}

### DOCUMENT TO PROCESS
"""


def get_extraction_prompt(document_text: str) -> str:
    """
    Constructs the final prompt by combining the system instructions 
    with the actual document text.
    
    Args:
        document_text: The extracted text from the PDF
    
    Returns:
        Complete prompt ready for LLM
    """
    return f"{EXTRACTION_SYSTEM_PROMPT}\n\n[BEGIN DOCUMENT TEXT]\n{document_text}\n[END DOCUMENT TEXT]"

def get_expected_format_prompt(document_text: str) -> str:
    """
    Create a completely flexible and generic prompt for any document type
    """
    return f"""
You are an expert data extraction engine. Extract ALL meaningful information from any document type into structured Key-Value pairs.

CORE EXTRACTION PRINCIPLES:
1. Identify and extract every piece of structured information
2. Create descriptive, logical field names for all data points
3. Preserve exact wording and formatting from the source
4. Extract dates, numbers, names, titles, requirements, specifications
5. Capture relationships and context between data elements

UNIVERSAL EXTRACTION APPROACH:
- Scan for any structured data: titles, headings, labeled information, lists, tables
- Extract key-value relationships wherever they exist
- Identify important entities: names, dates, numbers, locations, organizations
- Capture process steps, requirements, specifications, or instructions
- Note any hierarchical or categorical information
- Extract metadata: document type, purpose, scope, timeline

FIELD NAMING STRATEGY:
- Use clear, descriptive names that explain what the data represents
- Be specific: "Assignment Timeline" not just "Timeline"
- Include context: "Primary Objective" vs "Secondary Objective"
- Use consistent naming patterns within the document
- Make field names self-explanatory and professional

COMPREHENSIVE EXTRACTION:
- Extract ALL text that provides meaningful information
- Don't skip small details - they may be important
- Capture both explicit data and implied information
- Include any instructions, requirements, or specifications
- Note any conditional or optional elements

OUTPUT FORMAT (JSON only, no markdown):
{{
    "global_notes": "Brief description of document type and overall purpose",
    "entries": [
        {{"Key": "Document Type", "Value": "detected_document_type", "Comments": null}},
        {{"Key": "descriptive_field_name", "Value": "exact_extracted_value", "Comments": "additional_context_if_any"}},
        {{"Key": "another_field_name", "Value": "another_extracted_value", "Comments": null}},
        ...continue for ALL information found...
    ]
}}

EXTRACTION RULES:
- Extract values EXACTLY as they appear in the document
- Use null for Comments if no additional context exists
- Create logical, descriptive field names for every piece of data
- Include document type/purpose as the first entry
- Extract 100% of meaningful content - leave nothing out
- Maintain original formatting and terminology
- Group related information logically through field naming

DOCUMENT TEXT TO ANALYZE:
{document_text}
"""


def get_validation_prompt(json_output: str, original_text: str) -> str:
    """
    Optional: Create a validation prompt to check if extraction was complete
    
    Args:
        json_output: The JSON output from initial extraction
        original_text: Original PDF text
    
    Returns:
        Validation prompt for quality assurance
    """
    return f"""You are a data validation assistant.

TASK: Review the extracted data and confirm if ALL information from the original text has been captured.

ORIGINAL TEXT:
{original_text}

EXTRACTED JSON:
{json_output}

RESPONSE FORMAT:
{{
  "is_complete": true/false,
  "missing_items": ["list any missing data points"],
  "accuracy_notes": "any observations about fidelity"
}}

Output ONLY the JSON object."""
