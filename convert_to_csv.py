import pandas as pd

columns = [
    "ID", "Diagnosis",
    "radius1", "texture1", "perimeter1", "area1",
    "smoothness1", "compactness1", "concavity1",
    "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2",
    "smoothness2", "compactness2", "concavity2",
    "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3",
    "smoothness3", "compactness3", "concavity3",
    "concave_points3", "symmetry3", "fractal_dimension3"
]

# Read original dataset
df = pd.read_csv("wdbc.data", header=None)

# Assign column names
df.columns = columns

# Save as CSV
df.to_csv("test_data.csv", index=False)

print("CSV file created successfully!")
print("Dataset shape:", df.shape)
print(df.head())