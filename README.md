# PathoVision

PathoVision is a Streamlit-based medical diagnostics application that helps visualize and interpret pathology and biochemistry reports. It provides an interactive interface for clinicians and laboratories to analyze test results, generate insights, and produce patient summaries.

## Live Demo
https://pathovision-eqdbuq3aqftab6ck5xenfp.streamlit.app/

## Overview

The application allows users to upload medical reports such as CBC, LFT, RFT, Lipid Profile, and Blood Sugar tests. It automatically extracts data, visualizes trends, and provides AI-assisted interpretations. The goal is to simplify report analysis and improve diagnostic efficiency.

## Features

- Upload and parse pathology or biochemistry reports  
- Interactive data visualization using Plotly and Streamlit  
- Automated interpretation of test results  
- PDF report generation for patient summaries  
- Secure and easy deployment on Streamlit Cloud or locally

## Screenshots
<img width="1920" height="1200" alt="Screenshot (1424)" src="https://github.com/user-attachments/assets/630717ec-3605-431d-b771-ab735eef2f41" />

<img width="1821" height="793" alt="image" src="https://github.com/user-attachments/assets/c03e6ef6-2326-4586-b0c8-2f1a5f7f1cfb" />

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/684132d5-33fb-4e52-97e5-1362e523b202" />

<img width="1886" height="844" alt="image" src="https://github.com/user-attachments/assets/d2ce634e-56a1-4823-90db-050314c0b321" />

<img width="1919" height="1000" alt="image" src="https://github.com/user-attachments/assets/1839359f-6be4-4af5-b578-3a01cf430901" />

<img width="1417" height="911" alt="image" src="https://github.com/user-attachments/assets/ed494f06-c07f-47e4-92b6-97397208ddbd" />





## Technologies Used

- Python 3.14  
- Streamlit  
- Pandas  
- Plotly  
- Scikit-learn  
- Matplotlib  
- Docker (optional for deployment)  

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/kaszuk29-ship-it/Pathovision.git
   cd Pathovision
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

- Launch the app locally or visit the hosted version:  
  [https://pathovision-eqdbuq3aqftab6ck5xenfp.streamlit.app/](https://pathovision-eqdbuq3aqftab6ck5xenfp.streamlit.app/)  
- Upload your pathology or biochemistry report (PDF or image).  
- View automated analysis and visualizations.  
- Export the results as a PDF summary.  

## Supported Tests

- Complete Blood Count (CBC)  
- Liver Function Test (LFT)  
- Renal Function Test (RFT)  
- Lipid Profile  
- Blood Sugar (FBS)  

## Configuration

You can modify the `config.yaml` file to adjust reference ranges or thresholds for specific tests.  
To add new test types or AI models, extend the `models/` directory.  
Sample data files can be placed in the `data/` folder.

## Deployment

To deploy on Streamlit Cloud:
```
streamlit deploy
```

To deploy using Docker:
```
docker build -t pathovision .
docker run -p 8501:8501 pathovision
```

## Author
Kaviya shree V
