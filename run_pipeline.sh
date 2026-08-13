#!/bin/bash
echo "Step 1: EDA..."
python src/eda/eda.py
echo "Step 2: Preprocessing..."
python src/preprocessing/preprocessor.py
echo "Step 3: Training models..."
python src/models/scorecard.py
echo "Step 4: Fraud detection..."
python src/models/fraud_detector.py
echo "Step 5: Evaluation..."
python src/evaluation/evaluator.py
echo "Step 6: Risk strategy..."
python src/strategy/risk_strategy.py
echo "Pipeline complete. Run: streamlit run dashboard/app.py"
