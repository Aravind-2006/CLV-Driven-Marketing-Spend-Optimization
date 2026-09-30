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