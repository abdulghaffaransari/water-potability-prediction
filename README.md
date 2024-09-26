# Water Potability Prediction

## Project Description
This project leverages advanced machine learning techniques to develop predictive models that assess the potability of water based on various chemical, physical, and biological features. By analyzing comprehensive datasets, our models aim to identify safe drinking water sources, ultimately contributing to public health and environmental sustainability. The integration of MLflow and DVC allows for streamlined model tracking and evaluation, ensuring reproducibility and effective monitoring of model performance. Through this initiative, we aspire to drive informed decision-making regarding water quality management and safety.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://dagshub.com/abdulghaffaransari/water-potability-prediction.git
   cd water-potability-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure you have your data pre-processed and saved in the `./src/data/processed/` directory.
2. Update the `params.yaml` file with your model parameters.
3. Run the evaluation script:
   ```bash
   python evaluate.py
   ```

## Model Evaluation
The evaluation script assesses two machine learning models: Random Forest and Gradient Boosting. Metrics logged include accuracy, precision, recall, and F1-score, which are recorded using MLflow and DVC Live for real-time monitoring.

## Results
The evaluation results will be saved in `reports/metrics.json`. You can also view the metrics logged in MLflow for detailed analysis.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
