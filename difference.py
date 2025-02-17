import pandas as pd
import difflib

# Load the Excel file (Ensure xlrd is installed for .xls files)
file_path = "Copy of Mackly MC Vs Mackly Stock Rep.xls"
xls = pd.ExcelFile(file_path, engine="xlrd")  

# Read Sheet 1 with correct header row (Index 3 based on preview)
sheet1 = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=3)  

# Read Sheet 2 with correct header row (Index 2 based on preview)
sheet2 = pd.read_excel(xls, sheet_name=xls.sheet_names[1], header=2)  

# Clean column names (strip spaces and drop NaNs)
sheet1.columns = sheet1.columns.astype(str).str.strip()
sheet2.columns = sheet2.columns.astype(str).str.strip()

# Print column names to confirm correctness
print("\nCorrected Sheet 1 Columns:", sheet1.columns.tolist())
print("Corrected Sheet 2 Columns:", sheet2.columns.tolist())

# Define the unique ID column (Check if correct now)
unique_id_col = "ItemBarcode"  # Ensure this exists in both sheets

# Validate if the unique ID column exists
if unique_id_col not in sheet1.columns or unique_id_col not in sheet2.columns:
    close_matches1 = difflib.get_close_matches(unique_id_col, sheet1.columns, n=1, cutoff=0.5)
    close_matches2 = difflib.get_close_matches(unique_id_col, sheet2.columns.dropna(), n=1, cutoff=0.5)
    suggested_col = close_matches1[0] if close_matches1 else (close_matches2[0] if close_matches2 else None)
    raise KeyError(f"Unique ID column '{unique_id_col}' not found. Did you mean '{suggested_col}'?")

# Define columns to compare
columns_to_compare = ["CollectionNumber"]  

# Ensure all columns exist in both sheets
for col in columns_to_compare:
    if col not in sheet1.columns or col not in sheet2.columns:
        raise KeyError(f"Column '{col}' not found in one of the sheets.")

# Merge the sheets on the unique ID column to compare corresponding rows
merged = sheet1.merge(sheet2, on=unique_id_col, suffixes=('_sheet1', '_sheet2'))

# Create an empty list to store differences
differences = []

# Compare column values and store differences
for col in columns_to_compare:
    diff_rows = merged[merged[f"{col}_sheet1"] != merged[f"{col}_sheet2"]]
    for index, row in diff_rows.iterrows():
        differences.append({
            "Unique ID": row[unique_id_col],
            "Field": col,
            "current stock": row[f"{col}_sheet1"],
            "market stock": row[f"{col}_sheet2"]
        })

# Convert differences into a DataFrame
diff_df = pd.DataFrame(differences)

# Save differences to a new Excel file
output_file = "Differences.xlsx"
diff_df.to_excel(output_file, index=False)

print(f"Differences saved to {output_file}")
