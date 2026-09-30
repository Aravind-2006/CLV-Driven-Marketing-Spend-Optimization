# MLOps Pipeline for Customer Lifetime Value-Driven Marketing Spend Optimization

## Running with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t clv-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 --name clv-service clv-api
   ```

3. **Access API Documentation:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

## Data Versioning with DVC

Data versioning is managed using **DVC** (Data Version Control) with a local storage remote (`C:\Users\Arav\dvc-storage`).

- **Tracked Data Files:**
  - `data/Online Retail.xlsx` (UCI Online Retail raw purchase dataset, 541k rows)
  - `data/rfm_features.csv` (Computed customer Recency, Frequency, Monetary feature table, 4,339 rows)

- **Reproduce / Pull Data:**
  ```bash
  dvc pull
  ```

## Running the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Features included:
- **Technical Panel**: Live model metrics (MAE, R²), training timestamp, and actual vs. predicted monetary value chart.
- **Business Panel**: Customer segmentation distribution (High/Medium/Low), marketing spend allocation summary, cost savings vs naive baseline, and an interactive live CLV & spend calculator.

## Drift Detection

The project uses Evidently AI to monitor potential data drift between the reference customer dataset and simulated incoming customer data.

The reference dataset is `data/rfm_features.csv`. Since this student project does not have live production traffic, the current dataset is simulated by applying a realistic shift to the Recency feature to represent customers becoming less recently active.

The drift check compares:
- Recency
- Frequency
- Monetary

Run the drift check with:

```powershell
python src/drift_check.py

The generated HTML report is saved to:
reports/drift_report.html
Open reports/drift_report.html in a web browser to inspect the Evidently drift results.