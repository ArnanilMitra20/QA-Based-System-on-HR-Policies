import os
import requests
import json
import time
import random

# --- API Config ---
API_KEY = "AIzaSyBNQZNrz_0Pacf8vfaH_DSMFMTPuDa1VI8"
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

# --- Main Function ---
def generate_sectorwise_hr_data(sectors, sleep_between=2, max_retries=5):
    headers = {"Content-Type": "application/json"}

    for sector in sectors:
        print(f"\n📁 Generating HR policy data for sector: {sector}")

        full_prompt = (
            f"Generate a very detailed and comprehensive HR policy and practices document for a company in the {sector} sector.\n"
            f"Include the following sections with examples, measurable KPIs, and sub-policies wherever applicable:\n\n"
            f"1. Company Overview (fictitious name, size, location)\n"
            f"2. Lead HR Contact (name, designation, role)\n"
            f"3. Recruitment Strategy\n"
            f"   - Channels used, diversity hiring, tech tools used\n"
            f"4. Onboarding & Training\n"
            f"   - Duration, tools/platforms, mentorship\n"
            f"5. Performance Management\n"
            f"   - Appraisal frequency, review criteria, feedback loops\n"
            f"6. Remote/Flexible Work Policy\n"
            f"   - Eligibility, expectations, tools used\n"
            f"7. Employee Wellness & Mental Health\n"
            f"   - Initiatives, leave policies, burnout prevention\n"
            f"8. Diversity, Equity & Inclusion (DEI)\n"
            f"   - Goals, programs, cultural sensitivity\n"
            f"9. Learning & Development\n"
            f"   - Career growth, upskilling, certifications\n"
            f"10. Use of Technology in HR\n"
            f"    - HRIS tools, analytics platforms\n"
            f"11. Employee Engagement Programs\n"
            f"    - Events, recognition, surveys\n"
            f"12. Legal Compliance & Ethics\n"
            f"    - Labor laws, anti-harassment, grievance redressal\n"
            f"13. Compensation & Benefits\n"
            f"    - Pay bands, bonus structure, perks\n"
            f"14. Metrics & KPIs\n"
            f"    - Attrition rate, offer acceptance rate, engagement score\n"
            f"15. Challenges & Solutions specific to the {sector} sector\n\n"
            f"Format the response as a structured JSON object with proper section headers."
        )

        payload = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 2048  # Increase for more detailed output
            }
        }

        retries = 0
        while retries < max_retries:
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()

                text = ""
                for part in result.get("candidates", [])[0].get("content", {}).get("parts", []):
                    if "text" in part:
                        text += part["text"] + "\n"

                if not text:
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

        # Save the output
        filepath = os.path.join(OUTPUT_DIR, f"{sector}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"sector": sector, "hr_policy_data": text.strip()}, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved detailed HR policy for {sector} sector to {filepath}")
        time.sleep(sleep_between + random.uniform(0, 1))


# --- Run It ---
if __name__ == "__main__":
    generate_sectorwise_hr_data(sectors_list)
    print("\n🎉 All extended sector HR policy files saved in 'generated_hr_data/'")
