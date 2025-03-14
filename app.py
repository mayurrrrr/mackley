import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.title("Desired Style Sales Prediction App")
st.write("Upload your Excel file containing sales data, then type the desired style numbers and select a platform to forecast the next 3 months' sales.")

# --------------------------------------------------------------------
# Step A: Upload the Excel file and process the data
# --------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded_file is not None:
    # Load the data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()  # clean column names
    st.write("Data loaded successfully. Here's a preview:")
    st.dataframe(df.head())

    # -------------------------------
    # Step B: Fill Missing Data and Convert Numeric Columns
    # -------------------------------
    cat_cols = [
        "Item Name", "Collection No",
        # Removed "Style Number", "Size" from here so they remain normal columns
        "Gender", "Category", "Sub Category", "Sillhouette", "Print",
        "Print Technique", "Color Combo", "Fabric - Top", "Fabric - Bottom",
        "Fabric - Full Garment", "Platform"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    
    numeric_cols = ["GRN Qty", "Retail Price", "FOB", "consignment price", "QTY SOLD", "Month", "Year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # -------------------------------
    # Step C: Sort and Create Lag Features
    # -------------------------------
    df = df.sort_values(by=["ITEM", "Platform", "Year", "Month"])
    
    df['lag_1'] = df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(1).fillna(0)
    df['lag_2'] = df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(2).fillna(0)
    df['lag_3'] = df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(3).fillna(0)
    
    df['target_3m'] = (
        df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(-1).fillna(0) +
        df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(-2).fillna(0) +
        df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(-3).fillna(0)
    )
    
    st.write("Data after creating lag features:")
    st.dataframe(df.head(10))
    
    # -------------------------------
    # Step D: Feature Engineering & Encoding
    # -------------------------------
    # We remove "Style Number" and "Size" from cat_features so they remain as normal columns.
    cat_features = [
        "Sillhouette", "Print", "Print Technique", "Color Combo",
        "Fabric - Top", "Fabric - Bottom", "Fabric - Full Garment",
        "Gender", "Category", "Sub Category", "Collection No",
        # "Style Number", "Size" intentionally excluded
    ]
    
    # One-hot encode only the columns in cat_features
    df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)
    
    # Convert "Platform", "ITEM", "Item Name" to categorical
    for col in ["Platform", "ITEM", "Item Name"]:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype("category")
    
    # We'll keep "Style Number" and "Size" as normal columns
    # so we do NOT remove them from df_encoded in 'non_predictive'
    non_predictive = ["Barcode", "QTY SOLD", "target_3m", "Item Name", "Platform", "ITEM", "GRN Date"]
    features = [c for c in df_encoded.columns if c not in non_predictive]
    
    X = df_encoded[features]
    y = df_encoded["target_3m"]
    
    # Ensure that any remaining object columns in X are converted
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = pd.to_numeric(X[col], errors="raise")
            except Exception:
                X[col] = X[col].astype("category")
    
    st.write("Features used in the model:")
    st.write(features)
    
    # --------------------------------------------------------------------
    # Step E: Model Training & Saving (if not already trained)
    # --------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        enable_categorical=True
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Model trained. Test RMSE: {rmse:.2f}, Test MAE: {mae:.2f}")
    
    with open("my_xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.write("Trained model saved to my_xgb_model.pkl")
    
    # --------------------------------------------------------------------
    # Step F: Iterative Forecast for Desired Style Numbers
    # --------------------------------------------------------------------
    st.subheader("Forecasting")
    desired_styles_input = st.text_input("Enter Desired Style Numbers (comma separated)", "C20-INWR-15, C20-INWR-16, C20-INWR-17")
    desired_styles = [s.strip() for s in desired_styles_input.split(",") if s.strip()]
    selected_platform = st.selectbox("Select Platform for Forecast", options=sorted(df["Platform"].unique()))
    
    if st.button("Run Forecast"):
        # Create an item-details subset (without dropping duplicates)
        df_item_details = df[[
            "ITEM", "Platform", "Collection No", "Style Number", "Size", "Item Name",
            "Category", "Sillhouette", "Fabric - Top", "Fabric - Bottom", "Fabric - Full Garment", "Color Combo"
        ]]
        
        # Filter for the desired style numbers and selected platform
        desired_details = df_item_details[
            (df_item_details["Style Number"].isin(desired_styles)) &
            (df_item_details["Platform"] == selected_platform)
        ]
        
        if desired_details.empty:
            st.error("No records found for the selected styles on the selected platform.")
        else:
            # Merge on all four keys: ITEM, Platform, Style Number, and Size
            desired_encoded = pd.merge(
                df_encoded,
                desired_details[["ITEM", "Platform", "Style Number", "Size"]],
                on=["ITEM", "Platform", "Style Number", "Size"],
                how="inner"
            )
            
            # We'll forecast for every row that matches
            rows_desired = desired_encoded.copy()
            
            predictions_list = []
            for idx, row in rows_desired.iterrows():
                current_state = row.copy()
                monthly_preds = []
                for i in range(3):
                    X_sample = current_state[features].to_frame().T
                    X_sample = X_sample.astype(X_train.dtypes.to_dict())
                    pred = model.predict(X_sample)[0]
                    monthly_preds.append(pred)
                    current_state['lag_3'] = current_state['lag_2']
                    current_state['lag_2'] = current_state['lag_1']
                    current_state['lag_1'] = pred
                
                predictions_list.append({
                    "ITEM": row["ITEM"],
                    "Platform": row["Platform"],
                    "Style Number": row["Style Number"],
                    "Size": row["Size"],
                    "Feb_pred": monthly_preds[0],
                    "Mar_pred": monthly_preds[1],
                    "Apr_pred": monthly_preds[2],
                    "Total_3m": sum(monthly_preds)
                })
            
            results_df_desired = pd.DataFrame(predictions_list)
            for col in ["Feb_pred", "Mar_pred", "Apr_pred", "Total_3m"]:
                results_df_desired[col] = results_df_desired[col].round(0).clip(lower=0).astype(int)
            
            st.subheader("Forecast Results for Desired Items (All Matching Rows):")
            st.dataframe(results_df_desired)
            
            # Merge forecast results with desired_details to get full style info
            detailed_desired = pd.merge(
                results_df_desired,
                desired_details,
                on=["ITEM", "Platform", "Style Number", "Size"],
                how="left"
            )
            for col in ["Feb_pred", "Mar_pred", "Apr_pred", "Total_3m"]:
                detailed_desired[col] = detailed_desired[col].round(0).clip(lower=0).astype(int)
            
            st.subheader("Detailed Forecast for Desired Style Numbers (Including Size):")
            st.dataframe(detailed_desired[[
                "ITEM", "Item Name", "Collection No", "Style Number", "Size",
                "Sillhouette", "Fabric - Top", "Fabric - Bottom", "Fabric - Full Garment",
                "Feb_pred", "Mar_pred", "Apr_pred", "Total_3m"
            ]])
            
            # --------------------------------------------------------------------
            # Step G: Aggregate & Visualize by Style Number and Color Combo
            # --------------------------------------------------------------------
            if not detailed_desired.empty:
                style_sums_desired = detailed_desired.groupby(["Style Number", "Color Combo"], as_index=False).agg({
                    "Feb_pred": "sum",
                    "Mar_pred": "sum",
                    "Apr_pred": "sum"
                })
                style_sums_desired["Total_3m"] = (
                    style_sums_desired["Feb_pred"] +
                    style_sums_desired["Mar_pred"] +
                    style_sums_desired["Apr_pred"]
                )
                for col in ["Feb_pred", "Mar_pred", "Apr_pred", "Total_3m"]:
                    style_sums_desired[col] = style_sums_desired[col].round(0).clip(lower=0).astype(int)
                
                style_sums_desired["Label"] = style_sums_desired["Style Number"] + " (" + style_sums_desired["Color Combo"] + ")"
                
                st.subheader("Aggregated Forecast by Style Number and Color Combo:")
                st.write(style_sums_desired)
                
                import matplotlib.pyplot as plt
                import numpy as np
                x = np.arange(len(style_sums_desired))
                width = 0.8
                fig, ax = plt.subplots(figsize=(12,6))
                ax.bar(x, style_sums_desired['Feb_pred'], width, label='February')
                ax.bar(x, style_sums_desired['Mar_pred'], width,
                       bottom=style_sums_desired['Feb_pred'], label='March')
                ax.bar(x, style_sums_desired['Apr_pred'], width,
                       bottom=style_sums_desired['Feb_pred']+style_sums_desired['Mar_pred'], label='April')
                ax.set_xlabel("Style Number (Color Combo)")
                ax.set_ylabel("Predicted Sales (Next 3 Months)")
                ax.set_title("Aggregated Forecast by Style Number and Color Combo (Feb, Mar, Apr)")
                ax.set_xticks(x)
                ax.set_xticklabels(style_sums_desired['Label'], rotation=45)
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("Detailed forecast data is not available. Please run the forecast.")
            
            # --------------------------------------------------------------------
            # Step H: Save Detailed Forecast for Desired Items
            # --------------------------------------------------------------------
            output_file = "Desired_Style_Forecast_NoDrops_MEN.xlsx"
            detailed_desired.to_excel(output_file, index=False)
            st.download_button("Download Forecast Results", data=open(output_file, "rb").read(), file_name=output_file)
