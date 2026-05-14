# End-to-End GCP MLOps Architecture for Housing Price Prediction

# Author: Pedro Yanez Melendez

This project implements an end-to-end MLOps workflow on Google Cloud Platform for housing price prediction, integrating data engineering, machine learning model training, managed deployment, serverless inference, prediction logging, and model monitoring.

The objective of this repository is to demonstrate a production-oriented cloud architecture where a machine learning model is not only trained and deployed, but also exposed through an API, monitored through serving data, and validated through input and output drift detection.

A detailed 89-page Microsoft Word document is included in this repository with the full step-by-step technical explanation of the implementation, debugging process, deployment decisions, IAM configuration, monitoring setup, and final validation.

---

## Project Overview

The solution uses Google Cloud Platform services to build a complete machine learning lifecycle:

1. Load and prepare housing data.
2. Store raw and cleaned datasets in Cloud Storage.
3. Transform and analyze the data in BigQuery.
4. Train and evaluate multiple regression models in Python.
5. Export the final scikit-learn model artifact.
6. Deploy the model to Vertex AI.
7. Build a FastAPI service for online inference.
8. Containerize and deploy the API through Cloud Run.
9. Log predictions into BigQuery.
10. Prepare flattened serving data for monitoring.
11. Generate baseline predictions.
12. Configure Vertex AI Model Monitoring.
13. Validate input drift and output drift detection.

---

## Architecture

The main architecture follows this flow:

```text
Source Data
   ↓
Cloud Storage
   ↓
BigQuery
   ↓
Google Colab / Python ML Training
   ↓
Model Artifacts
   ↓
Vertex AI Model Registry & Endpoint
   ↓
FastAPI Service
   ↓
Cloud Run
   ↓
Vertex AI Online Prediction
   ↓
BigQuery Prediction Logs
   ↓
Flattened Serving Table
   ↓
Vertex AI Model Monitoring
```

The deployment pipeline also includes:

```text
FastAPI App
   ↓
Docker Image
   ↓
Cloud Build
   ↓
Artifact Registry
   ↓
Cloud Run
```

---

## Main GCP Services Used

- Google Cloud Storage
- BigQuery
- Vertex AI
- Vertex AI Model Registry
- Vertex AI Endpoint
- Vertex AI Model Monitoring
- Cloud Run
- Cloud Build
- Artifact Registry
- IAM
- Logs Explorer

---

## Machine Learning Workflow

The model was developed in Python using a structured scikit-learn pipeline.

### Dataset

The project uses housing data with the following original features:

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude
- MedHouseVal

The prediction target is:

```text
MedHouseVal
```

---

## Feature Engineering

Additional engineered features were created to improve the model input representation:

```text
RoomsPerBedroom = AveRooms / AveBedrms
EstimatedHouseholds = Population / AveOccup
GeoInteraction = Latitude * Longitude
```

These features were later required to align training, online prediction, batch prediction, and monitoring schemas.

---

## Model Training

Several regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- HistGradientBoostingRegressor

The final selected model was based on cross-validation performance and then tuned using randomized hyperparameter search.

---

## Final Model Metrics

The final model achieved the following test metrics:

```text
R²   = 0.84800
MAE  = 0.29350
RMSE = 0.45031
```

The target variable is expressed in units of 100,000 USD.  
For example, a prediction of `3.57` represents approximately `357,000 USD`.

---

## Vertex AI Deployment

The trained model was exported as a `joblib` artifact and deployed through Vertex AI.

A key technical challenge was ensuring compatibility between the local scikit-learn training environment and the Vertex AI serving container.

The model was retrained using compatible versions:

```text
scikit-learn = 1.2.2
numpy        = 1.26.4
```

The final artifact was uploaded as:

```text
model.joblib
```

---

## Vertex AI Compatibility Fix

The original pipeline depended on pandas DataFrame column names.  
However, Vertex AI scikit-learn serving expects numeric array-style payloads for online prediction.

To solve this, the pipeline was adapted to use positional NumPy-based inputs.

The preprocessing step was updated to use numeric feature indexes instead of column names:

```python
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, list(range(len(numeric_features))))
])
```

The model now expects payloads with 11 numeric values in the same order used during training.

---

## Online Prediction Payload

Example request format:

```json
{
  "instances": [
    [3.5, 10, 5, 1, 1000, 3, 34, -118, 5.0, 333.3, -4012]
  ]
}
```

Example successful response:

```json
{
  "predictions": [1.932731900076394]
}
```

---

## FastAPI and Cloud Run

A FastAPI application was created to wrap the Vertex AI endpoint and expose a `/predict` API.

The API was containerized with Docker and deployed to Cloud Run.

### Main API responsibilities

- Receive prediction requests.
- Validate the request structure.
- Forward the payload to Vertex AI.
- Return the model prediction.
- Log successful predictions to BigQuery.

---

## Containerization and Deployment

The FastAPI service was containerized using Docker.

The image was built with Cloud Build and pushed to Artifact Registry.

The final deployment path used Artifact Registry instead of the legacy `gcr.io` format:

```text
LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY/IMAGE_NAME
```

Key deployment components:

- `main.py`
- `requirements.txt`
- `Dockerfile`
- Cloud Build
- Artifact Registry
- Cloud Run

---

## BigQuery Prediction Logging

Each successful prediction was logged into BigQuery using a table named:

```text
housing_analytics.prediction_logs
```

The logging table included fields such as:

- timestamp
- instances
- prediction
- status

This enabled the creation of serving data for model monitoring.

---

## Flattened Serving Table

Vertex AI Model Monitoring requires serving data in a structured format where input features are available as separate columns.

For that reason, the original logged prediction table was transformed into a flattened table:

```text
housing_analytics.prediction_logs_flat
```

This table included:

- timestamp
- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude
- RoomsPerBedroom
- EstimatedHouseholds
- GeoInteraction
- predictions

---

## Model Monitoring

Vertex AI Model Monitoring was configured to evaluate:

- Input feature drift
- Output prediction drift

The monitoring workflow used:

- Training baseline data
- Serving prediction logs
- Flattened BigQuery serving table
- Batch prediction baseline
- Vertex AI monitoring run

---

## Input Drift Validation

The input drift monitoring run succeeded after restructuring the serving data into a flattened format.

Several features exceeded the configured drift threshold, confirming that the monitoring pipeline could detect distribution changes between training data and serving data.

Example drift scores observed:

```text
AveOccup = 0.9663
AveRooms = 0.6966
MedInc   = 0.5736
```

These values were based on synthetic test calls, so they validate the monitoring mechanism rather than proving real production model degradation.

---

## Output Drift Validation

Output drift was validated after creating baseline predictions through batch inference.

The batch prediction job processed:

```text
20,640 rows
0 failures
```

A clean baseline prediction table was created in BigQuery and used as the baseline for Vertex AI Model Monitoring.

The output drift result showed:

```text
Prediction drift score = 0.5766
Baseline mean          = 2.07
Target mean            = 3.38
```

Because the serving rows were synthetic API tests, this result confirms that output drift monitoring works technically.

---

## Main Technical Challenges Solved

This project included several practical cloud engineering and MLOps challenges:

- Aligning project ID, project number, and resource paths.
- Migrating from `gcr.io` assumptions to Artifact Registry.
- Creating the correct Artifact Registry repository.
- Granting Cloud Build permissions to push Docker images.
- Granting Cloud Run permissions to call Vertex AI.
- Granting Cloud Run permissions to write BigQuery logs.
- Debugging Cloud Run HTTP 500 errors through Logs Explorer.
- Fixing Vertex AI payload nesting issues.
- Aligning local training format with Vertex AI serving format.
- Flattening BigQuery serving logs for Model Monitoring.
- Creating batch prediction baselines for output drift monitoring.
- Validating both input drift and output drift.

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- FastAPI
- Docker
- Google Cloud Storage
- BigQuery
- Vertex AI
- Vertex AI Endpoints
- Vertex AI Model Monitoring
- Cloud Run
- Cloud Build
- Artifact Registry
- IAM
- Logs Explorer
- SQL
- MLOps

---

## Repository Structure

A suggested repository structure is:

```text
.
├── README.md
├── GCP_Project_python_colab.ipynb
├── GCP_Project.docx
├── main.py
├── requirements.txt
├── Dockerfile
├── artifacts/
│   └── model.joblib
└── docs/
    └── architecture-diagram.png
```

---

## Final Result

The final validated MLOps workflow includes:

- A trained regression model.
- A compatible Vertex AI model artifact.
- A deployed Vertex AI endpoint.
- A FastAPI inference layer.
- A containerized Cloud Run API.
- BigQuery prediction logging.
- Flattened serving data for monitoring.
- Baseline predictions generated through batch inference.
- Input drift monitoring.
- Output drift monitoring.

This project demonstrates a complete cloud-based MLOps implementation, connecting model development, deployment, API serving, observability, and monitoring into a single integrated architecture.
