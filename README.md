# Voltage-Prediction

Goal: Predict the next BatteryVoltage using Time and BatteryCurrent as input features in a deep learning model (LSTM).

Steps:
1. Data Preparation: The dataset was preprocessed by normalizing Time, BatteryCurrent, and BatteryVoltage. A sliding window approach (window size = 5) was used to prepare the data for the LSTM model.
Model Training: An LSTM model was trained to predict the next BatteryVoltage using past values of Time and BatteryCurrent.

2. Error Handling: A ValueError occurred due to mismatched feature dimensions during scaling. The MinMaxScaler expected 3 features but was given only 2 during prediction.

3. Solution: The prediction function was updated to handle this issue by adding a dummy BatteryVoltage column during scaling and inverse scaling only the predicted voltage.

4. Prediction Code: The revised code predicts BatteryVoltage using a trained LSTM model by scaling the input correctly and applying inverse scaling after prediction.
