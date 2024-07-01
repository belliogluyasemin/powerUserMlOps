# Power User Prediction, CI/CD Process, and Deployment to Cloud Run

![Diagram](attachment:deneme_diagram.drawio (2).png)

## Power User Prediction Steps

In this project, power users were identified through a detailed process involving feature engineering, model selection, and evaluation. The methodology used for identifying power users is outlined below:

### 1. Feature Engineering
- **Data Augmentation**: Additional features were created, such as the average number of products in a basket, to enrich the dataset and improve model performance.
- **Target Column**: Initially, power users were identified based on a z-score method. However, this method resulted in very few power users, making it insufficient. Therefore, based on the distributions, users who spent more than $110 were identified as power users, forming the basis for the classification target.

### 2. Model Training and Oversampling
- **Training Dataset Preparation**: The training dataset was adjusted using various oversampling methods to handle class imbalances.
- **Oversampling Techniques**: Methods like RandomOverSample, SMOTE, and ADASYN were employed to balance the dataset, ensuring that the models could effectively learn from both the majority and minority classes.

### 3. Model Selection
- **Hyperparameter Optimization**: GridSearchCV was used for hyperparameter tuning to find the best model configurations.
- **Comparison of Models**: Several models were compared based on their performance metrics, including KNN, XGBoost, Logistic Regression, and Random Forest. The XGBoost model with ADASYN oversampling showed the best performance.

| Model                                       | Recall  | Precision | Log Loss |
|---------------------------------------------|---------|-----------|----------|
| KNN                                         | 51.65%  | 88.70%    | 6.88%    |
| KNN_RandomOverSample                        | 47.25%  | 23.50%    | 15.45%   |
| KNN_SMOTE                                   | 62.64%  | 19.40%    | 19.13%   |
| KNN_ADASYN                                  | 28.57%  | 35.60%    | 24.31%   |
| XGBoost                                     | 71.43%  | 100.00%   | 5.23%    |
| XGBoost_RandomOverSample                    | 71.43%  | 55.60%    | 2.70%    |
| XGBoost_SMOTE                               | 71.43%  | 97.00%    | 1.41%    |
| **XGBoost_ADASYN**                          | 71.43%  | 100.00%   | 1.33%    |
| Logistic Regression                         | 67.03%  | 89.70%    | 1.21%    |
| Logistic Regression_RandomOverSample        | 78.02%  | 10.50%    | 14.46%   |
| Logistic Regression_SMOTE                   | 76.92%  | 11.60%    | 12.64%   |
| Logistic Regression_ADASYN                  | 84.62%  | 6.40%     | 24.08%   |
| RandomForest                                | 70.33%  | 100.00%   | 4.49%    |
| RandomForest_RandomOverSample               | 63.74%  | 87.90%    | 5.12%    |
| RandomForest_SMOTE                          | 72.53%  | 42.00%    | 3.16%    |
| RandomForest_ADASYN                         | 74.73%  | 32.70%    | 4.43%    |

### 4. Threshold Adjustment and Model Evaluation
- **Threshold Tuning**: The threshold value of the model was adjusted to optimize performance. Despite testing various thresholds, the default value of 0.5 was retained as it provided the best balance between recall and precision.
- **ROC AUC Curve Analysis**: The performance of the XGBoost ADASYN model was further validated using the ROC AUC curve, demonstrating strong predictive capabilities.

### 5. Create and Push Docker Image to Docker Hub

1. **Build the Docker image**: Use the Dockerfile provided to create an image tagged as `xgboost_adasyn_poweruser_image:v1`.
    ```sh
    docker build -t xgboost_adasyn_poweruser_image:v1 .
    ```

2. **Run the image locally**: Verify that the image was built correctly by running it locally.
    ```sh
    docker run -d -p 8000:8000 --name xgboost_container xgboost_adasyn_poweruser_image:v1
    ```

3. **Tag the image**: Before pushing the image to Docker Hub, tag it appropriately with your Docker Hub username and repository name.
    ```sh
    docker tag xgboost_adasyn_poweruser_image:v1 yaseminbellioglu/xgboost_adasyn_poweruser_image:v1
    ```

4. **Push the image**: Upload the image to your Docker Hub repository, making it available for deployment on other machines.
    ```sh
    docker push yaseminbellioglu/xgboost_adasyn_poweruser_image:v1
    ```

## 6. Cloud Run Deployment With CI/CD Process

Cloud Run is used because it provides automatic scaling, simple deployment, and cost efficiency by only charging for actual usage. Unlike VMs, it eliminates the need for manual management and maintenance, allowing for easier integration with other Google Cloud services.

### 1. Push Docker Image to Artifact Registry

First, create an Artifact Registry repository to store the Docker image. Follow these steps to push your Docker image to Artifact Registry:

1. **Authenticate with Google Cloud**: Ensure you are logged in to your Google Cloud account.
    ```sh
    gcloud auth login
    ```

2. **Set Your Project**: Configure your project settings.
    ```sh
    gcloud config set project PROJECT_ID
    ```

3. **Configure Docker**: Use the gcloud command-line tool to authenticate requests to Artifact Registry.
    ```sh
    gcloud auth configure-docker us-central1-docker.pkg.dev
    ```

4. **Build and Tag Your Docker Image**: Build the Docker image and tag it with the appropriate name.
    ```sh
    docker build -t xgboost_adasyn_poweruser_image:v1 .
    docker tag xgboost_adasyn_poweruser_image:v1 us-central1-docker.pkg.dev/psychic-root-424207-s9/xgboost/xgboost_adasyn_poweruser_image:cloudingv1
    docker push us-central1-docker.pkg.dev/psychic-root-424207-s9/xgboost/xgboost_adasyn_poweruser_image:cloudingv1
    ```

### 2. IAM Permissions

Before setting up the CI/CD pipeline, ensure that the service account used for GitHub Actions has the following IAM roles assigned:

- Artifact Registry Administrator
- Artifact Registry Writer
- BigQuery Admin
- Cloud Run Admin
- Editor
- Secret Manager Secret Accessor
- Service Account User

You can assign these roles in the Google Cloud Console under IAM & Admin > IAM by editing the permissions for your service account.

### 3. CI/CD Process

The CI/CD process for PowerUser CR is managed using GitHub Actions. The workflow is defined in the `.github/workflows/docker-image.yml` file and includes steps for building, testing, and deploying the application.

### Workflow File

Here is the complete workflow file:

```yaml
name: CI/CD 
on:
  push:
    branches: [ main ]

jobs:
  build_and_test:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Repo  ##Checkout the Code: GitHub Actions checks out the source code from the GitHub repository
      uses: actions/checkout@v2

    - name: Set up Python ##Set up Python: Python 3.11 is set up
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
      
    - name: Install dependencies ##Install Dependencies: Required Python packages are installed
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
      
    - name: Run tests ##Run Tests: The application's tests are executed
      run: |
        python -m pytest  

  deploy:
    runs-on: ubuntu-latest
    needs: build_and_test
    steps:
    - name: Checkout Repo ##Checkout the Code: The source code is checked out again for the deployment job.
      uses: actions/checkout@v2

    - name: Set up gcloud CLI ##Set up gcloud CLI: gcloud CLI is set up for Google Cloud authentication
      uses: google-github-actions/auth@v1
      with:
        project_id: ${{ secrets.PROJECT_ID }}
        credentials_json: ${{ secrets.CREDENTIALS_JSON }}

    - name: Build and push container image ##Build and Push Docker Image: The Docker image is built and pushed to Artifact Registry
      env:
        PROJECT_ID: ${{ secrets.PROJECT_ID }}
      run: |
        gcloud auth configure-docker us-central1-docker.pkg.dev
        docker build -t us-central1-docker.pkg.dev/${PROJECT_ID}/xgboost/xgboost_adasyn_poweruser_image:cloudingv1 .
        docker push us-central1-docker.pkg.dev/${PROJECT_ID}/xgboost/xgboost_adasyn_poweruser_image:cloudingv1

    - name: Deploy to Cloud Run ##Deploy to Cloud Run: The application is deployed to Cloud Run with the necessary settings.
      run: |
        gcloud run deploy xgboost \
        --image=us-central1-docker.pkg.dev/${{ secrets.PROJECT_ID }}/xgboost/xgboost_adasyn_poweruser_image:cloudingv1 \
        --allow-unauthenticated \
        --port=8000 \
        --service-account=${{ secrets.SERVICE_ACCOUNT }} \
        --max-instances=5 \
        --region=us-central1 \
        --project=${{ secrets.PROJECT_ID }}
```



### 4. Access the API from CLoud Run

Use the provided URL to interact with your API. The URL will look something like this:









