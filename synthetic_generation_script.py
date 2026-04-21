import pandas as pd
import numpy as np

def generate_full_synthetic_csv(filename='synthetic_job_data_v2.csv', num_samples=1000):
    np.random.seed(42)

    # 1. Text Probabilities for Naive Bayes Validation
    vocab = ["salary", "experience", "wire", "immediate", "management", "logistics"]
    true_prob_legit = [0.30, 0.40, 0.01, 0.05, 0.15, 0.09]
    true_prob_fraud = [0.10, 0.05, 0.40, 0.35, 0.05, 0.05]

    # 2. Generate 20% Fraudulent Labels
    is_fraud = (np.random.rand(num_samples) < 0.20).astype(int)

    data_rows = []
    for i in range(num_samples):
        label = is_fraud[i]
        
        # --- Generate Text for Columns 7, 8, and 9 ---
        probs = true_prob_fraud if label == 1 else true_prob_legit
        counts = np.random.multinomial(15, probs / np.sum(probs))
        text_list = []
        for word, count in zip(vocab, counts):
            text_list.extend([word] * count)
        np.random.shuffle(text_list)

        chunk = len(text_list) // 3
        description = " ".join(text_list[:chunk])
        requirements = " ".join(text_list[chunk:2*chunk])
        benefits = " ".join(text_list[2*chunk:])

        # --- Generate Metadata Features (Probabilistic) ---
        # Fraudulent jobs are often remote and lack logos
        telecommuting = 1 if (label == 1 and np.random.rand() < 0.8) else (0 if np.random.rand() < 0.8 else 1)
        has_logo = 0 if (label == 1 and np.random.rand() < 0.7) else 1
        has_questions = 1 if np.random.rand() > 0.5 else 0

        row = {
            "job_id": i + 1,
            "title": "Data Entry Specialist" if label == 1 else "Hardware Engineer",
            "location": "Remote" if label == 1 else "Medford, MA",
            "department": "Finance" if label == 1 else "ECE",
            "salary_range": "High Weekly Pay" if label == 1 else "80k-120k",
            "company_profile": "Secure Hires Inc" if label == 1 else "Tufts University",
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "telecommuting": telecommuting,
            "has_company_logo": has_logo,
            "has_questions": has_questions,
            "employment_type": "Contract" if label == 1 else "Full-time",
            "required_experience": "Not Applicable" if label == 1 else "Mid-Senior level",
            "required_education": "High School" if label == 1 else "Bachelor's Degree",
            "industry": "Staffing" if label == 1 else "Education",
            "function": "Admin" if label == 1 else "Engineering",
            "fraudulent": label
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    # Ensure columns follow the exact requested order
    column_order = [
        "job_id", "title", "location", "department", "salary_range", 
        "company_profile", "description", "requirements", "benefits", 
        "telecommuting", "has_company_logo", "has_questions", 
        "employment_type", "required_experience", "required_education", 
        "industry", "function", "fraudulent"
    ]
    df = df[column_order]
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {num_samples} rows.")

if __name__ == "__main__":
    generate_full_synthetic_csv()