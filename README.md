# Vehicle Platoon Behaviour Prediction

A machine learning project for predicting and analyzing vehicle platoon behavior, dynamics, and interactions in autonomous and semi-autonomous driving scenarios.

## Project Overview

This project develops predictive models to understand and forecast vehicle platooning behavior, including vehicle spacing, acceleration patterns, and coordination dynamics. It combines data preprocessing, feature engineering, and machine learning techniques to provide insights into platoon formation and driver behavior.

## Project Structure

```
vehicle-platoon-behaviour-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/                    # Original, immutable raw data
│   ├── processed/              # Cleaned and processed data
│   └── sample/                 # Sample datasets for testing
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Initial data analysis
│   ├── 02_feature_engineering.ipynb   # Feature creation
│   ├── 03_model_training.ipynb        # Model development
│   └── 04_visualization.ipynb         # Results visualization
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning and transformation
│   ├── feature_engineering.py  # Feature extraction and engineering
│   ├── platoon_extraction.py   # Platoon identification logic
│   ├── train_model.py          # Model training pipeline
│   ├── evaluate_model.py       # Model evaluation metrics
│   ├── visualization.py        # Plotting and visualization utilities
│   └── utils.py                # Helper functions
│
├── models/
│   ├── random_forest.pkl       # Trained model artifact
│   └── scaler.pkl              # Feature scaler artifact
│
├── outputs/
│   ├── plots/                  # Generated visualizations
│   ├── confusion_matrix/       # Model evaluation plots
│   └── reports/                # Analysis reports
│
├── dashboard/
│   └── app.py                  # Interactive dashboard application
│
└── docs/
    ├── architecture.png        # System architecture diagram
    ├── methodology.md          # Project methodology documentation
    └── project_report.pdf      # Comprehensive project report
```

## Features

- **Data Preprocessing**: Cleaning, validation, and transformation of vehicle trajectory data
- **Feature Engineering**: Extraction of behavioral and kinematic features for platoon analysis
- **Platoon Extraction**: Automatic identification and segmentation of vehicle platoons
- **Machine Learning Models**: Random Forest and other algorithms for behavior prediction
- **Visualization Tools**: Comprehensive plots for data exploration and results analysis
- **Interactive Dashboard**: Web-based application for real-time visualization and analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shreshta312/vehicle-platoon-behaviour-prediction.git
cd vehicle-platoon-behaviour-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Feature Engineering
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### Model Training
```bash
python src/train_model.py
```

### Model Evaluation
```bash
python src/evaluate_model.py
```

### Run Dashboard
```bash
python dashboard/app.py
```

## Notebooks

- **01_data_exploration.ipynb**: Exploratory data analysis (EDA) of vehicle trajectory datasets
- **02_feature_engineering.ipynb**: Feature creation and selection process
- **03_model_training.ipynb**: Model development and hyperparameter tuning
- **04_visualization.ipynb**: Results visualization and analysis

## Key Modules

- `data_preprocessing.py`: Data cleaning and normalization
- `feature_engineering.py`: Feature extraction from raw trajectory data
- `platoon_extraction.py`: Logic for identifying vehicle platoons
- `train_model.py`: Model training pipeline with cross-validation
- `evaluate_model.py`: Comprehensive model evaluation and metrics
- `visualization.py`: Plotting utilities for data and results
- `utils.py`: General utility functions

## Models

- **Random Forest Model** (random_forest.pkl): Trained classifier for platoon behavior prediction
- **Feature Scaler** (scaler.pkl): StandardScaler for feature normalization

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- [Add other specific requirements]

## Results & Outputs

- Confusion matrices and classification reports in `outputs/confusion_matrix/`
- Generated visualizations in `outputs/plots/`
- Analysis reports in `outputs/reports/`

## Methodology

See `docs/methodology.md` for detailed information about:
- Data collection and preprocessing approach
- Feature engineering strategy
- Model architecture and selection
- Evaluation metrics and validation approach

## Architecture

Refer to `docs/architecture.png` for a visual representation of the project architecture and data flow.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Author

**Shreshta** - [GitHub Profile](https://github.com/shreshta312)

## Contact

For questions or inquiries about this project, please feel free to open an issue or reach out.

## References

[Add relevant papers, datasets, and resources here]

## Acknowledgments

- Dataset sources
- Collaborators and advisors
- Research papers and methodologies that inspired this work

---

**Last Updated**: May 2026
