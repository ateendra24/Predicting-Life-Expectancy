# Life Expectancy Prediction using IBM Watson AutoAI

This project utilizes IBM Watson Studio's AutoAI capabilities to analyze a global Life Expectancy dataset. The objective is to automatically preprocess the data, select relevant features, train multiple machine learning models, and identify the best-performing pipeline for predicting life expectancy. AutoAI streamlines the process by handling data cleaning, algorithm selection, and hyperparameter optimization. The final model is evaluated for accuracy and deployed as a REST API for real-time predictions.

## 🎯 Project Overview

The project focuses on predicting life expectancy based on various health, social, and economic factors across different countries. Using IBM Watson Studio's AutoAI, we automatically generate and compare multiple machine learning pipelines to find the optimal solution for this regression problem.

## Demo Video
[Demo Video](https://drive.google.com/file/d/1wnATCC5DFYS7TK-qTcnRFwn1ZQEFPrCK/view?usp=sharing "Demo Video")


### 🌍 Interactive Web Interface

For easy access to life expectancy predictions without technical knowledge, we provide a user-friendly web interface:

**🔗 [Life Expectancy Predictor Web App](https://life-expectancy-prediction-green.vercel.app)**


## 📊 Dataset Features

The dataset includes the following key features for life expectancy prediction:

- **Country**: Geographic location
- **Status**: Development status (Developed/Developing)
- **Adult Mortality**: Adult mortality rates (probability of dying between 15 and 60 years per 1000 population)
- **Infant Deaths**: Number of infant deaths per 1000 population
- **Alcohol**: Alcohol consumption recorded per capita (15+) consumption (in litres)
- **Percentage Expenditure**: Expenditure on health as a percentage of GDP per capita (%)
- **Hepatitis B**: Hepatitis B immunization coverage among 1-year-olds (%)
- **Measles**: Number of reported cases per 1000 population
- **BMI**: Average Body Mass Index of entire population
- **Under-five Deaths**: Number of under-five deaths per 1000 population
- **Polio**: Polio immunization coverage among 1-year-olds (%)
- **Total Expenditure**: General government expenditure on health as a percentage of total government expenditure (%)
- **Diphtheria**: Diphtheria tetanus toxoid and pertussis immunization coverage among 1-year-olds (%)
- **HIV/AIDS**: Deaths per 1000 live births HIV/AIDS (0-4 years)
- **GDP**: Gross Domestic Product per capita (in USD)
- **Population**: Population of the country
- **Thinness 1-19 years**: Prevalence of thinness among children and adolescents for Age 10 to 19 (%)
- **Thinness 5-9 years**: Prevalence of thinness among children for Age 5 to 9 (%)
- **Income Composition of Resources**: Human Development Index in terms of income composition of resources
- **Schooling**: Number of years of schooling

## 🔧 Technology Stack

- **IBM Watson Studio**: AutoAI platform for automated machine learning
- **Python 3.11**: Primary programming language
- **Scikit-learn 1.3**: Machine learning library
- **LightGBM 4.2**: Gradient boosting framework (final model)
- **autoai-libs**: IBM's AutoAI transformation libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 🏗️ Project Structure

```
project/
├── _P4 - LGBM Regressor_ Predicting Life Expectancy.ipynb  # Main notebook
├── README.md                                                # Project documentation
└── screenshots/                                            # Project screenshots
    ├── 1. Reading Dataset.png
    ├── 2. Generating Pipelines.png
    ├── 3. Pipelines.png
    ├── 4. Progress.png
    ├── 5. Pipeline Comparison.png
    └── 6. API EndPoint.png
```

## 🚀 Getting Started

### Prerequisites

1. IBM Cloud account with Watson Studio access
2. Python 3.11 or higher
3. Required Python packages (see Installation section)

### Installation

Install the required packages using pip:

```bash
pip install ibm-watsonx-ai
pip install autoai-libs~=2.0
pip install scikit-learn==1.3.*
pip install -U lale~=0.8.3
pip install lightgbm==4.2.*
```

### Setup

1. **Watson Studio Configuration**:
   - Set up your IBM Cloud API key
   - Configure project ID and deployment URL
   - Ensure proper authentication credentials

2. **Data Connection**:
   - Upload your life expectancy dataset to Watson Studio
   - Configure data asset connection references

3. **Environment**:
   - Set appropriate CPU count for parallel processing
   - Configure memory and computational resources

## 📈 Model Pipeline

The AutoAI pipeline includes the following stages:

### 1. Data Preprocessing
- **Column Selection**: Automated feature selection
- **Missing Value Handling**: Imputation strategies for numerical and categorical data
- **String Compression**: Efficient encoding of categorical variables
- **Data Type Conversion**: Proper formatting of mixed data types

### 2. Feature Engineering
- **Mathematical Transformations**: Square root transformations for non-negative features
- **Feature Combinations**: Automatic generation of composite features
- **Feature Selection**: Intelligent selection of most relevant features
- **Scaling**: Optional standardization based on data characteristics

### 3. Model Training
- **Algorithm**: LightGBM Regressor with DART boosting

### 4. Model Evaluation
- **Scoring Metric**: Negative Root Mean Squared Error (RMSE)
- **Cross-validation**: Automated validation strategy
- **Holdout Testing**: 10% of data reserved for final evaluation

## 📊 Results

The final LightGBM model demonstrates strong performance in predicting life expectancy:

- **Model Type**: Gradient Boosting Regressor
- **Optimization Metric**: Negative Root Mean Squared Error
- **Holdout Split**: 90% training, 10% testing
- **Feature Engineering**: Automated with square root transformations and feature combinations

## 🌐 Deployment

The trained model can be deployed as a REST API endpoint:

1. **Model Storage**: Automated storage in Watson ML repository
2. **Deployment Space**: Configuration for production environment
3. **Online Scoring**: Real-time prediction capabilities
4. **API Testing**: Validation with sample data

### Sample API Usage

```python
import pandas as pd

# Prepare scoring payload
scoring_payload = {
    "input_data": [{
        'values': pd.DataFrame(test_data)
    }]
}

# Score the model
predictions = client.deployments.score(deployment_id, scoring_payload)
```


#### Features:
- **Simple Form Interface**: Enter country details, health indicators, and socioeconomic factors
- **Real-time Predictions**: Get instant life expectancy predictions
- **Input Validation**: Automatic validation of input ranges and data types
- **Visual Results**: Charts and graphs showing prediction confidence and contributing factors
- **Comparison Tool**: Compare predictions across different countries or scenarios
- **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices

#### How to Use:
1. Visit the web application link above
2. Fill in the required fields:
   - Select your country or region
   - Enter health metrics (BMI, mortality rates, disease rates)
   - Provide socioeconomic data (GDP, education, healthcare expenditure)
   - Input demographic information
3. Click "Predict Life Expectancy"
4. View your personalized prediction with explanatory insights

#### Input Fields:
The web interface includes user-friendly forms for all 20 model features:
- Country selection dropdown
- Development status (Developed/Developing)
- Health indicators (mortality rates, vaccination coverage)
- Economic factors (GDP, healthcare expenditure)
- Social metrics (education years, income composition)
- Demographic data (population, age-specific health metrics)

## 📸 Screenshots

The `screenshots/` directory contains visual documentation of the project workflow:

1. **Reading Dataset**: Data loading and initial exploration
2. **Generating Pipelines**: AutoAI pipeline generation process
3. **Pipelines**: Overview of generated pipeline options
4. **Progress**: Training progress and model evaluation
5. **Pipeline Comparison**: Comparison of different pipeline performances
6. **API Endpoint**: Deployed model API interface

![Pipelines.png](screenshots/3.%20Pipelines.png "Pipelines.png")

## 🔍 Key Features

- **Automated ML Pipeline**: Complete automation from data preprocessing to model deployment
- **Feature Engineering**: Intelligent feature creation and selection
- **Hyperparameter Optimization**: Automated tuning for optimal performance
- **Model Interpretability**: Clear pipeline structure and feature importance
- **Production Ready**: Direct deployment to Watson ML for real-time predictions
- **Scalable Architecture**: Configurable for different computational resources

## 🛠️ Configuration

The notebook is configured with the following settings:

- **CPU Utilization**: Dynamic CPU count detection
- **Memory Management**: Optimized for large datasets
- **Scoring Strategy**: Cross-validation with holdout testing
- **Pipeline Optimization**: Multiple algorithms compared automatically

## 📚 Usage Instructions

1. **Open the Notebook**: Launch `_P4 - LGBM Regressor_ Predicting Life Expectancy.ipynb`
2. **Setup Credentials**: Enter your IBM Cloud API key when prompted
3. **Configure Environment**: Ensure proper project and space IDs
4. **Run Pipeline**: Execute cells sequentially for complete workflow
5. **Evaluate Results**: Review model performance metrics
6. **Deploy Model**: Optional deployment for production use



## 📄 License

This project is subject to IBM Watson Studio licensing terms and the International License Agreement for Non-Warranted Programs.


---

**Note**: This project showcases automated machine learning capabilities and demonstrates best practices for regression problems using IBM's enterprise AI platform.
