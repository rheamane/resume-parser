import pandas as pd
import re
import spacy
import json
import os

# Load English model
nlp = spacy.load("en_core_web_sm")

# Input/output
input_csv = "data\Resume.csv"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# Helper functions
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else None


def extract_phone(text):
    match = re.search(r"(\+\d{1,3}[- ]?)?\d{10}", text)
    return match.group(0) if match else None


def extract_linkedin(text):
    match = re.search(r"(https?:\/\/)?(www\.)?linkedin\.com\/[A-Za-z0-9_\-/]+", text)
    return match.group(0) if match else None


def extract_section(text, section_name):
    """Find text under a section like EXPERIENCE, EDUCATION, etc."""
    pattern = rf"{section_name}[\s\S]+?(?=\n[A-Z ]{{3,}}|\Z)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).strip() if match else None


# Read dataset
df = pd.read_csv(input_csv)
# Testing on first 5 rows
df = df.head(5)

for _, row in df.iterrows():
    text = str(row["Resume_str"])

    parsed_resume = {
        "ID": row["ID"],
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "LinkedIn": extract_linkedin(text),
        "Summary": extract_section(text, "SUMMARY"),
        "Experience": extract_section(text, "EXPERIENCE"),
        "Education": extract_section(text, "EDUCATION"),
        "Skills": extract_section(text, "SKILLS"),
        "Projects": extract_section(text, "PROJECTS"),
        "Category": row["Category"],
    }

    output_path = os.path.join(output_dir, f"{row['ID']}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_resume, f, indent=4, ensure_ascii=False)

print(f"✅ Parsed {len(df)} resumes and saved JSON files in {output_dir}/")
