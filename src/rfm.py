import pandas as pd

# read the raw dataset
df = pd.read_excel("data/Online Retail.xlsx")

# just checking - print first 5 rows to see what it looks like
print(df.head())
# remove rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# remove returns/cancellations (negative quantity)
df = df[df["Quantity"] > 0]

print("Rows after cleaning:", len(df))
import datetime as dt

# add a "total price" column for each row (Quantity x UnitPrice)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# pick a reference date - one day after the last purchase in the dataset
reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

# group by customer, calculate R, F, M
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",                                     # Frequency
    "TotalPrice": "sum"                                         # Monetary
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print(rfm.head())
print("Total customers:", len(rfm))
rfm.to_csv("data/rfm_features.csv")
print("Saved to data/rfm_features.csv")