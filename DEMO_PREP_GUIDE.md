# 🎙️ Interview Demo Preparation Guide

## 🎯 **The Goal**
You need to demonstrate that you didn't just "do the assignment" — you built a **scalable, intelligent product**. Your focus should be on the **"No Hardcoding"** rule and the **Data Fidelity**.

---

## 🛠️ **1. Pre-Flight Checklist (1 Hour Before)**

### **Environment Setup**
- [ ] **Clean Desktop:** Close unnecessary tabs/apps. Hide messy desktop icons.
- [ ] **Code Editor:** Open VS Code. Close all files except:
    - `backend.py` (The brain)
    - `prompt_templates.py` (The AI logic)
    - `ai-doc-processor-frontend/src/App.tsx` (The UI)
- [ ] **Terminal:** Have two terminals ready and **already running**:
    - Terminal 1: `python backend.py` (Backend)
    - Terminal 2: `npm run dev` (Frontend)
- [ ] **Browser:** Open `localhost:5173` (or your local port).
- [ ] **Data:** Have `Data Input.pdf` handy on your Desktop for easy drag-and-drop.

### **Backup Plan (Crucial)**
- [ ] **Pre-processed Output:** Have a copy of a perfect `Output.xlsx` generated earlier. If the live demo fails (API error, slow internet), say: *"The API seems slow right now, but here is the output I generated just before the call."*
- [ ] **Screenshots/Video:** If everything crashes, have screenshots of the UI ready.

---

## 🗣️ **2. The Demo Script (10-15 Minutes)**

### **Phase 1: The "Elevator Pitch" (2 Minutes)**
*Start with the problem, not the code.*

> "Hi! For this assignment, I focused on the core challenge: **Generic Extraction**. Most parsers break if you change the PDF format. My solution uses a **Dynamic Key Extraction** engine. It doesn't look for specific words like 'Invoice Date'; instead, it reads the document context to *decide* what the columns should be. This ensures 100% data fidelity without hardcoding."

### **Phase 2: The Live Demo (5 Minutes)**
*Share your screen. Show the UI.*

1.  **Upload:** Drag `Data Input.pdf` into the dropzone.
2.  **The "Magic" Moment:** Point out the **Real-time Logs**.
    > "You can see here the system is chunking the PDF. It's not just regex matching; it's sending context to the LLM."
3.  **The Result:** When the table appears:
    > "Notice the column names. I didn't define these in code. The system realized this document needed 'Event Name', 'Time', and 'Speaker' columns automatically."
4.  **Deduplication (Bonus):**
    > "I noticed raw LLM output often has duplicates. I implemented a custom deduplication logic that reduced redundancy by ~60% in my tests."
5.  **Export:** Click "Download Excel" and open it to show the clean formatting.

### **Phase 3: Code Walkthrough (5 Minutes)**
*Switch to VS Code. Don't scroll aimlessly. Jump to specific files.*

1.  **Show `backend.py`:**
    > "This is the FastAPI entry point. It handles the upload and orchestrates the pipeline."
2.  **Show `prompt_templates.py` (The Winner):**
    > "This is the most important part. Instead of asking for specific JSON keys, I ask the LLM to 'analyze the document structure and extract a list of objects'. This fulfills the 'No Hardcoding' requirement."
3.  **Show `utils_dedup.py` (The Extra Mile):**
    > "This is where I handle data quality. I use fuzzy matching to merge similar rows."

---

## ❓ **3. Anticipated Questions & Answers**

**Q: How do you ensure you didn't hardcode keys?**
**A:** *"I use a two-step prompting strategy. First, the LLM analyzes the chunk to understand the schema. Then, it extracts data according to that dynamic schema. You won't find a list of column names anywhere in my python files."*

**Q: How does it handle large PDFs?**
**A:** *"I implemented a chunking strategy. The PDF is split into manageable text blocks (e.g., 4000 characters) with overlap to ensure we don't cut off context. Each chunk is processed independently, then merged."*

**Q: Why did you choose React/FastAPI?**
**A:** *"FastAPI is the gold standard for Python AI backends because of its async capabilities and automatic documentation. React offers the best component-based architecture for building interactive dashboards like this."*

**Q: What would you improve?**
**A:** *"I'd add a 'Human-in-the-loop' feature where users can correct the column names before the final export, and the system learns from those corrections."* (You actually have the foundation for this!)

---

## 🧠 **4. Key Phrases to Use**
- "Dynamic Schema Generation"
- "Context-Aware Extraction"
- "Deterministic Output" (trying to make the AI consistent)
- "Production-Ready Architecture"
