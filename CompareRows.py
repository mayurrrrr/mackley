import streamlit as st
import pandas as pd
import difflib
import re
from io import BytesIO

# Function to normalize text values (remove spaces, special characters, lowercase)
def normalize_text(value):
    if isinstance(value, str):  # Check if value is a string
        return re.sub(r'[^a-zA-Z0-9-]', '', value).strip().lower()  # Keep hyphens (-)
    return value  # Return non-string values as they are

# Set Streamlit UI title
st.title("Excel Comparison Tool")

# Choose comparison mode
comparison_mode = st.radio("Choose Comparison Mode:", ("Compare Two Sheets in One File", "Compare Two Separate Excel Files"))

if comparison_mode == "Compare Two Sheets in One File":
    # Upload a single Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])

    if uploaded_file:
        # Load Excel file
        xls = pd.ExcelFile(uploaded_file)

        # Select sheets
        sheet_names = xls.sheet_names
        st.write("Available Sheets:", sheet_names)

        sheet1_name = st.selectbox("Select First Sheet", sheet_names, index=0)
        sheet2_name = st.selectbox("Select Second Sheet", sheet_names, index=1)

        # Choose header row for each sheet
        header1_row = st.number_input(f"Select header row for {sheet1_name}", min_value=0, max_value=50, value=0, step=1)
        header2_row = st.number_input(f"Select header row for {sheet2_name}", min_value=0, max_value=50, value=0, step=1)

        # Read sheets with user-defined headers
        sheet1 = pd.read_excel(xls, sheet_name=sheet1_name, engine="openpyxl" if uploaded_file.name.endswith("xlsx") else "xlrd", header=int(header1_row))
        sheet2 = pd.read_excel(xls, sheet_name=sheet2_name, engine="openpyxl" if uploaded_file.name.endswith("xlsx") else "xlrd", header=int(header2_row))

elif comparison_mode == "Compare Two Separate Excel Files":
    # Upload two separate Excel files
    uploaded_file1 = st.file_uploader("Upload First Excel File", type=["xls", "xlsx"], key="file1")
    uploaded_file2 = st.file_uploader("Upload Second Excel File", type=["xls", "xlsx"], key="file2")

    if uploaded_file1 and uploaded_file2:
        # Choose header row for each file
        header1_row = st.number_input("Select header row for First Excel File", min_value=0, max_value=50, value=0, step=1, key="header1")
        header2_row = st.number_input("Select header row for Second Excel File", min_value=0, max_value=50, value=0, step=1, key="header2")

        # Load Excel files with user-defined headers
        sheet1 = pd.read_excel(uploaded_file1, engine="openpyxl" if uploaded_file1.name.endswith("xlsx") else "xlrd", header=int(header1_row))
        sheet2 = pd.read_excel(uploaded_file2, engine="openpyxl" if uploaded_file2.name.endswith("xlsx") else "xlrd", header=int(header2_row))

# Ensure both sheets or files are loaded
if 'sheet1' in locals() and 'sheet2' in locals():
    # Clean column names (remove spaces, special characters, and lowercase everything)
    sheet1.columns = sheet1.columns.astype(str).str.strip().str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()
    sheet2.columns = sheet2.columns.astype(str).str.strip().str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()

    # Apply the normalization function to all values in the sheets
    #sheet1 = sheet1.applymap(normalize_text)
    #sheet2 = sheet2.applymap(normalize_text)

    # Display column names
    st.write(f"Sheet 1 Columns: {sheet1.columns.tolist()}")
    st.write(f"Sheet 2 Columns: {sheet2.columns.tolist()}")

    # Select Unique ID Column
    unique_id_col = st.selectbox("Select Unique ID Column", sheet1.columns.intersection(sheet2.columns))

    # Select columns to compare
    columns_to_compare = st.multiselect("Select Columns to Compare", sheet1.columns.intersection(sheet2.columns))

    if unique_id_col and columns_to_compare:
        # Merge sheets/files on unique ID
        merged = sheet1.merge(sheet2, on=unique_id_col, suffixes=('_sheet1', '_sheet2'))

        # Debugging: Print column names
        st.write("Merged DataFrame Columns:", merged.columns.tolist())

        # Store differences
        differences = []

        # Compare column values
        for col in columns_to_compare:
            diff_rows = merged[merged[f"{col}_sheet1"] != merged[f"{col}_sheet2"]]
            for index, row in diff_rows.iterrows():
                differences.append({
                    "Unique ID": row[unique_id_col],
                    "Field": col,
                    "Sheet 1 Value": row[f"{col}_sheet1"],
                    "Sheet 2 Value": row[f"{col}_sheet2"]
                })

        # Convert to DataFrame
        diff_df = pd.DataFrame(differences)

        # Show results
        if not diff_df.empty:
            st.write("### Differences Found:")
            st.dataframe(diff_df)

            # Create a button to download only when clicked
            if st.button("Generate Differences File"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    diff_df.to_excel(writer, index=False)
                output.seek(0)

                st.download_button("Download Differences File", output, file_name="Differences.xlsx", mime="application/vnd.ms-excel")
        else:
            st.success("No differences found!")
