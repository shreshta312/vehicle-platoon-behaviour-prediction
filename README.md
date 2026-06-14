# Safety-Aware Vehicle Platoon Behaviour Prediction

This project builds a machine learning pipeline to classify vehicle platoon behaviours such as **FOLLOW**, **BRAKE**, **CHANGE_LANE_LEFT**, and **CHANGE_LANE_RIGHT** using trajectory-based driving features.

The project includes controlled platoon scenario generation, deep learning models, temporal modelling, adversarial robustness testing, SHAP explainability, and an extension toward real-world NGSIM highway trajectory data.

## Project Motivation

Vehicle platooning is an important concept in intelligent transportation systems where vehicles travel in coordinated groups to improve safety, traffic flow, and fuel efficiency.

In platoon-like driving, predicting whether a vehicle should follow, brake, or change lanes is safety-critical. This project focuses on building an ML pipeline for such behaviour prediction using surrounding vehicle features.

## Behaviour Classes

| Label | Behaviour |
|---|---|
| 0 | BRAKE |
| 1 | FOLLOW |
| 2 | CHANGE_LANE_LEFT |
| 3 | CHANGE_LANE_RIGHT |

## Features Used

The model uses trajectory-based features such as:

- Ego vehicle velocity
- Front vehicle distance
- Front relative velocity
- Time gap
- Left lane front/rear distance
- Right lane front/rear distance

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- SHAP
- Joblib

## Dataset

The main pipeline uses a controlled synthetic platoon dataset generated for safety-aware behaviour prediction.

The project also includes an experimental NGSIM extension, where real-world highway trajectory data is cleaned and converted into platoon-like features such as front vehicle distance, relative velocity, time gap, and adjacent-lane distances.

Raw NGSIM files are not included because of size limitations.

## Model Results

| Model | Accuracy | Notes |
|---|---:|---|
| Baseline Deep Learning Model | 94.67% | Initial behaviour classification model |
| Safety-Aware Model | 94.00% | Included adversarial training concepts |
| Temporal LSTM Model | 94.00% | Used sequential driving features |

The safety-aware model achieved approximately **97% BRAKE recall** on the controlled platoon dataset, which is important for safety-critical driving decisions.

## Explainability

SHAP was used to interpret model predictions. For BRAKE predictions, the most important features included:

- Front relative velocity
- Front distance
- Time gap
- Ego velocity

This shows that the model learned physically meaningful driving patterns.

## Robustness Testing

FGSM-based adversarial testing was used to evaluate model stability under small perturbations in input features. The safety-aware model showed improved robustness under stronger perturbations, though some safety-critical class confusions remain future work.

## Project Structure

```text
vehicle-platoon-behaviour-prediction/
│
├── data/
│   ├── external/
│   └── processed/
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── simulation/
│   └── webots/
│
├── src/
│   ├── data/
│   ├── data_generation/
│   ├── explainability/
│   ├── models/
│   ├── safety/
│   └── visualization/
│
├── requirements.txt
├── .gitignore
└── README.md