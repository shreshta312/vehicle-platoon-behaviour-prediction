# Safety-Aware Vehicle Platoon Behaviour Prediction

This project builds a machine learning pipeline to predict vehicle platoon behaviours such as **FOLLOW**, **BRAKE**, **CHANGE_LANE_LEFT**, and **CHANGE_LANE_RIGHT** using trajectory-based driving features.

The project combines controlled platoon scenario generation, deep learning models, adversarial robustness testing, temporal modelling, SHAP explainability, and an initial extension toward real-world NGSIM highway trajectory data.

---

## Project Motivation

Vehicle platooning is an important concept in intelligent transportation systems where vehicles travel in coordinated groups to improve traffic flow, fuel efficiency, and road safety.
In such systems, predicting whether a vehicle should continue following, brake, or change lanes is safety-critical.

This project focuses on building a behaviour prediction pipeline that can classify platoon-like driving decisions using surrounding vehicle information such as:

* Ego vehicle speed
* Front vehicle distance
* Relative velocity
* Time gap
* Adjacent lane front and rear distances

---

## Key Features

* Generated synthetic platoon-like driving scenarios for controlled ML training
* Trained baseline deep learning model for behaviour classification
* Built a safety-aware model using adversarial training concepts
* Evaluated safety-critical BRAKE recall
* Added temporal modelling using MLP and LSTM-based approaches
* Performed FGSM adversarial robustness testing
* Used SHAP explainability to interpret model predictions
* Extended the pipeline toward real-world NGSIM trajectory data
* Engineered front-vehicle and adjacent-lane features from noisy highway trajectory data

---

## Behaviour Classes

The model predicts one of four vehicle behaviours:

| Label | Behaviour         |
| ----- | ----------------- |
| 0     | BRAKE             |
| 1     | FOLLOW            |
| 2     | CHANGE_LANE_LEFT  |
| 3     | CHANGE_LANE_RIGHT |

---

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* SHAP
* Joblib

---

## Dataset

This project uses two types of datasets:

### 1. Synthetic Platoon Dataset

A controlled synthetic dataset was generated to simulate platoon-like driving scenarios with features such as front distance, relative velocity, time gap, and adjacent lane distances.

This dataset was used to build and test the main behaviour prediction pipeline.

### 2. NGSIM Trajectory Extension

The project also includes an experimental real-world extension using NGSIM highway trajectory data.
The raw NGSIM data was cleaned and transformed into platoon-like driving features by extracting:

* Front vehicle distance
* Front relative velocity
* Time gap
* Adjacent lane front/rear distances
* Derived behaviour labels from trajectory changes

The NGSIM extension is included as an ongoing real-world validation direction.

---

## Model Results

### Controlled Platoon Scenario Results

| Model                        | Accuracy | Notes                                      |
| ---------------------------- | -------: | ------------------------------------------ |
| Baseline Deep Learning Model |   94.67% | Initial behaviour classification model     |
| Safety-Aware Model           |   94.00% | Included adversarial training concepts     |
| Temporal LSTM Model          |   94.00% | Used sequential driving behaviour features |

The safety-aware model achieved strong performance on safety-critical braking behaviour, with **BRAKE recall of approximately 97%** on the controlled platoon dataset.

---

## Explainability

SHAP was used to interpret model predictions and identify the most important features influencing safety-critical decisions.

For BRAKE predictions, the most important features included:

* Front relative velocity
* Front distance
* Time gap
* Ego velocity

This shows that the model learned physically meaningful driving features rather than relying only on arbitrary patterns.

---

## Robustness Testing

The project includes FGSM-based adversarial testing to evaluate how stable the model remains under small perturbations in input features.

The robustness experiments showed that the safety-aware model reduced overall attack success under stronger perturbations, although some safety-critical class confusions still require further improvement.

---

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
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic platoon data

```bash
python src/data_generation/generate_platoon_data.py
```

### 3. Train baseline model

```bash
python src/models/train_baseline_model.py
```

### 4. Train safety-aware model

```bash
python src/models/train_safe_model.py
```

### 5. Evaluate model

```bash
python src/models/evaluate_model.py
```

### 6. Run adversarial robustness testing

```bash
python src/safety/adversarial_evaluation.py
```

### 7. Run SHAP explainability

```bash
python src/explainability/run_shap_fast.py
```

---

## NGSIM Extension

The NGSIM pipeline is included to show how real-world highway trajectory data can be converted into platoon-like behaviour prediction features.

Example command:

```bash
python src/data/prepare_ngsim_dataset_v2.py --input data/raw/ngsim.csv --sample 500000
python src/models/train_ngsim_rf_model.py
```

Raw NGSIM files are not included in this repository due to size constraints.

---

## Current Limitations

* The primary high-accuracy results are based on controlled synthetic platoon scenarios.
* NGSIM labels are derived from observed trajectory changes, not manually annotated driver intentions.
* Real-world trajectory prediction is noisier and requires richer temporal and neighbouring vehicle features.
* The simulation component is experimental and not a full autonomous driving simulator.

---

## Future Work

* Improve real-world NGSIM behaviour prediction using better temporal feature extraction
* Add platoon-specific filtering for vehicle-following scenarios
* Build a cleaner simulation demo for visualizing predicted behaviours
* Experiment with sequence models such as LSTM/GRU/Transformer architectures
* Add targeted adversarial training for safety-critical class confusions
* Improve lane-change prediction using richer neighbouring vehicle context

---

## Summary

This project demonstrates a complete ML workflow for safety-aware vehicle platoon behaviour prediction, including dataset generation, model training, evaluation, robustness testing, explainability, and real-world trajectory feature engineering.
