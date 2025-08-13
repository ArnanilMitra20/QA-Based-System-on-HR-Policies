import os
import requests
import json
import time
import random
import re

# --- API Config ---
API_KEY = "AIzaSyDCNh2JK9WOePuq4EXKs9F33hVXvfnmRCA"  # <-- Replace with your own key
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# --- Output Folder ---
OUTPUT_DIR = "generated_hr_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Extended Sectors List ---
sectors_list = [
    "Technology", "Healthcare", "Finance", "Retail", "Manufacturing",
    "Education", "Telecommunications", "Energy", "Transportation", "Hospitality",
    "E-commerce", "Pharmaceuticals", "Logistics", "Insurance", "Legal Services",
    "Aerospace", "Agriculture", "Construction", "Entertainment", "NGO and Nonprofit"
]

def clean_json_text(raw_text: str) -> str:
    """
    Removes markdown code fences and trims extra spaces so we can parse JSON safely.
    """
    cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^```", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()

def generate_sectorwise_hr_data(sectors, sleep_between=2, max_retries=5):
    headers = {"Content-Type": "application/json"}

    for sector in sectors:
        print(f"\n📁 Generating HR policy data for sector: {sector}")

        # Explicit strict JSON schema instruction
        full_prompt = f"""
You are a JSON-only generator.  
Your task is to output *only* valid JSON — no explanations, no markdown code fences, no extra text.  

You must strictly follow this schema, filling all fields with detailed and realistic HR policies for a company in the "{sector}" sector:

{{
  "company_overview": {{
    "name": "string",
    "size": "string",
    "location": "string"
  }},
  "lead_hr_contact": {{
    "name": "string",
    "designation": "string",
    "role": "string"
  }},
  "recruitment_strategy": {{
    "channels": ["string"],
    "diversity_hiring": "string",
    "tools_used": ["string"]
  }},
  "onboarding_and_training": {{
    "duration": "string",
    "platforms": ["string"],
    "mentorship_program": "string"
  }},
  "performance_management": {{
    "appraisal_frequency": "string",
    "criteria": ["string"],
    "feedback_loops": "string"
  }},
  "remote_flexible_work_policy": {{
    "eligibility": "string",
    "expectations": "string",
    "tools_used": ["string"]
  }},
  "employee_wellness_and_mental_health": {{
    "initiatives": ["string"],
    "leave_policies": "string",
    "burnout_prevention": "string"
  }},
  "dei": {{
    "goals": ["string"],
    "programs": ["string"],
    "cultural_sensitivity": "string"
  }},
  "learning_and_development": {{
    "career_growth": "string",
    "upskilling": ["string"],
    "certifications": ["string"]
  }},
  "technology_in_hr": {{
    "hris_tools": ["string"],
    "analytics_platforms": ["string"]
  }},
  "employee_engagement_programs": {{
    "events": ["string"],
    "recognition_methods": ["string"],
    "surveys": "string"
  }},
  "legal_compliance_and_ethics": {{
    "labor_laws": ["string"],
    "anti_harassment_policy": "string",
    "grievance_redressal": "string"
  }},
  "compensation_and_benefits": {{
    "pay_bands": "string",
    "bonus_structure": "string",
    "perks": ["string"]
  }},
  "metrics_and_kpis": {{
    "attrition_rate": "string",
    "offer_acceptance_rate": "string",
    "engagement_score": "string"
  }},
  "sector_specific_challenges_and_solutions": "string"
}}

Rules:
1. Do NOT include triple backticks or markdown formatting.
2. Do NOT add explanations or comments — only output the JSON object above.
3. Ensure the JSON is syntactically valid and matches the schema exactly.
"""


        payload = {
            "contents": [{"parts": [{"text": full_prompt.strip()}]}],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 2048
            }
        }

        retries = 0
        text = ""
        while retries < max_retries:
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()

                for part in result.get("candidates", [])[0].get("content", {}).get("parts", []):
                    if "text" in part:
                        text += part["text"] + "\n"

                if not text.strip():
                    text = "No content returned."
                break

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    wait = 2 ** retries + random.uniform(0, 1)
                    print(f"  ⏳ Rate limit hit. Retrying in {wait:.2f} seconds...")
                    time.sleep(wait)
                    retries += 1
                else:
                    print(f"  ❌ HTTP Error: {e}")
                    text = f"ERROR: {e}"
                    break
            except Exception as e:
                print(f"  ❌ Other Error: {e}")
                text = f"ERROR: {e}"
                break
        else:
            text = f"ERROR: Max retries exceeded for sector {sector}"

        # --- Clean & Parse JSON ---
        parsed_data = None
        try:
            cleaned_text = clean_json_text(text)
            parsed_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Output for {sector} is not valid JSON, saving as plain text.")

        # --- Save to file ---
        filepath = os.path.join(OUTPUT_DIR, f"{sector}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            if parsed_data:
                json.dump({"sector": sector, "generated_hr_data": parsed_data}, f, indent=2, ensure_ascii=False)
            else:
                json.dump({"sector": sector, "generated_hr_data": text.strip()}, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved detailed HR policy for {sector} sector to {filepath}")
        time.sleep(sleep_between + random.uniform(0, 1))

if __name__ == "__main__":
    generate_sectorwise_hr_data(sectors_list)
    print("\n🎉 All extended sector HR policy files saved in 'generated_hr_data/'")
