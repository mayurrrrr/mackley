import streamlit as st
import pandas as pd
from io import BytesIO

# Function to read Excel files with automatic engine selection
def read_excel_file(uploaded_file, header_row):
    try:
        uploaded_file.seek(0)  # Reset file pointer
        file_name = uploaded_file.name.lower()

        # Detect file type and use the correct engine
        if file_name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl", header=int(header_row))
        elif file_name.endswith(".xls"):
            return pd.read_excel(uploaded_file, engine="xlrd", header=int(header_row))
        else:
            st.error("❌ Invalid file format. Please upload a valid Excel file (.xls or .xlsx).")
            return None
    except Exception as e:
        st.error(f"❌ Error reading file {uploaded_file.name}: {e}")
        return None

# Function to preview raw data before setting headers
def preview_excel_file(uploaded_file):
    try:
        uploaded_file.seek(0)  # Reset file pointer
        file_name = uploaded_file.name.lower()

        # Detect file type and use the correct engine
        if file_name.endswith(".xlsx"):
            df_preview = pd.read_excel(uploaded_file, engine="openpyxl", header=None)  # Read all rows
        elif file_name.endswith(".xls"):
            df_preview = pd.read_excel(uploaded_file, engine="xlrd", header=None)  # Read all rows
        else:
            st.error("❌ Invalid file format. Please upload a valid Excel file (.xls or .xlsx).")
            return None

        return df_preview.head(5)  # Show first 5 rows for preview
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Function to compare two dataframes based on a selected column
def compare_dataframes(df1, df2, column_name):
    if column_name not in df1.columns or column_name not in df2.columns:
        st.error(f"❌ Column '{column_name}' not found in both files.")
        return None

    # Find missing rows in the secondary file compared to the primary file
    primary_values = set(df1[column_name].dropna().unique())
    secondary_values = set(df2[column_name].dropna().unique())
    missing_values = primary_values - secondary_values

    # Create a dataframe of missing rows
    missing_rows = df1[df1[column_name].isin(missing_values)]
    return missing_rows

# Function to convert dataframe to CSV for download
def convert_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Function to convert dataframe to Excel for download
def convert_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Missing Rows")
    output.seek(0)
    return output

# Set Streamlit UI title
st.title("🔍 Excel Header Preview, Selection & Comparison")

# Upload two separate Excel files
uploaded_file1 = st.file_uploader("📂 Upload Primary Excel File", type=["xls", "xlsx"], key="file1")
uploaded_file2 = st.file_uploader("📂 Upload Secondary Excel File", type=["xls", "xlsx"], key="file2")

if uploaded_file1:
    # Show raw preview for primary file
    st.write("### 🔍 Primary File Preview (First 5 Rows - Select Header)")
    preview_df1 = preview_excel_file(uploaded_file1)
    if preview_df1 is not None:
        st.dataframe(preview_df1)

    header1_row = st.number_input("🔍 Select header row for Primary File", min_value=0, max_value=50, value=0, key="header1")
    df1 = read_excel_file(uploaded_file1, header1_row)

    if df1 is not None:
        st.write("✅ **Headers in Primary File (After Selection):**", df1.columns.tolist())

if uploaded_file2:
    # Show raw preview for secondary file
    st.write("### 🔍 Secondary File Preview (First 5 Rows - Select Header)")
    preview_df2 = preview_excel_file(uploaded_file2)
    if preview_df2 is not None:
        st.dataframe(preview_df2)

    header2_row = st.number_input("🔍 Select header row for Secondary File", min_value=0, max_value=50, value=0, key="header2")
    df2 = read_excel_file(uploaded_file2, header2_row)

    if df2 is not None:
        st.write("✅ **Headers in Secondary File (After Selection):**", df2.columns.tolist())

# Compare columns and find missing rows
if uploaded_file1 and uploaded_file2 and df1 is not None and df2 is not None:
    st.write("### 🔄 Compare Columns and Find Missing Rows")

    # Select a column to compare
    common_columns = list(set(df1.columns).intersection(set(df2.columns)))
    if common_columns:
        selected_column = st.selectbox("🔍 Select a column to compare", common_columns)
        if selected_column:
            missing_rows = compare_dataframes(df1, df2, selected_column)
            if missing_rows is not None:
                st.write(f"### 🚨 Missing Rows in Secondary File (Compared to Primary File)")
                st.write(f"Total Missing Rows: **{len(missing_rows)}**")
                st.dataframe(missing_rows)

                # Download options
                st.write("### ⬇️ Download Missing Rows")
                download_format = st.radio("Select download format", ["CSV", "Excel"])

                if download_format == "CSV":
                    csv_data = convert_to_csv(missing_rows)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="missing_rows.csv",
                        mime="text/csv",
                    )
                elif download_format == "Excel":
                    excel_data = convert_to_excel(missing_rows)
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name="missing_rows.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
    else:
        st.error("❌ No common columns found between the two files.")