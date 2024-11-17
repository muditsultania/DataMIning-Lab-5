import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset with realistic ranges
num_samples = 500
data = {
    "CGPA": np.round(np.random.uniform(6, 10, num_samples), 2),
    "GRE": np.random.randint(280, 340, num_samples),
    "GMAT": np.random.randint(500, 800, num_samples),
    "TOEFL": np.random.randint(80, 120, num_samples),
    "Research_Pubs": np.random.randint(0, 5, num_samples),  # Number of research articles
    "Mini_Project": np.random.choice([0, 1], num_samples),  # Binary: 0 or 1
    "Internship": np.random.choice([0, 1], num_samples),  # Binary: 0 or 1
}

# Calculate 'Admit' probability based on synthetic logic
admit_chance = (
    0.3 * data["CGPA"]
    + 0.2 * (data["GRE"] / 340)
    + 0.2 * (data["GMAT"] / 800)
    + 0.1 * (data["TOEFL"] / 120)
    + 0.1 * data["Research_Pubs"]
    + 0.05 * data["Mini_Project"]
    + 0.05 * data["Internship"]
)
# Convert probability to binary outcome
data["Admit"] = (admit_chance > 2.5).astype(int)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save dataset to CSV
dataset_path = "university_admission_data.csv"
df.to_csv(dataset_path, index=False)
print(f"Dataset saved to: {dataset_path}")

# Display a preview of the dataset
df.head()
