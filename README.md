# Bike-Sharing System Analysis and Forecasting

## Overview
This repository provides an end-to-end analysis of a bike-sharing system, focusing on **demand forecasting** and **trip duration analysis**. By leveraging advanced machine learning models, it offers actionable insights to optimize operations and enhance customer experience.

## Dataset
The analysis is based on two datasets:
1. **Demand Data**:
   - Features: Date, Hour, Demand, Weather Conditions, Public Holiday, Working Day.
   - Objective: Forecast hourly bike demand.
2. **Duration Data**:
   - Features: Duration, Date, Borough, Weather Conditions, Public Holiday, Working Day.
   - Objective: Analyze trip durations and identify key factors affecting them.

## Objective
1. **Demand Forecasting**:
   - Predict hourly bike demand to improve resource allocation.
2. **Trip Duration Analysis**:
   - Identify factors affecting trip durations and suggest operational improvements.

## Methodology
1. **Data Preprocessing**:
   - Cleaned and prepared data for analysis.
   - Engineered features like weather conditions, public holidays, and weekdays.

2. **Demand Forecasting**:
   - Models Used: ARIMA, SARIMA, and LSTM.
   - Evaluated model performance using R² and RMSE metrics.
   - Forecasted demand for future months.

3. **Duration Analysis**:
   - Models Used: Random Forest, Gradient Boosting, and Neural Networks.
   - Analyzed feature importance to identify key factors affecting durations.
   - Clustering and segmentation for borough-level insights.

4. **Performance Evaluation**:
   - Compared models across multiple train-test splits (70:30, 60:40, 80:20).
   - Captured metrics (MAE, RMSE, R²) and saved predictions for further analysis.
     
## Future Work
- **Incorporate Real-Time Data**:
  - Use live weather and event data to improve forecasting accuracy.
- **Advanced Models**:
  - Experiment with Transformer-based models and NeuralProphet for forecasting.
- **Integration**:
  - Build a dashboard for real-time visualization and decision-making.

## How to Use
  **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

All Rights Reserved

Copyright (c) 2024 [Kayalks]

This repository and its contents, including but not limited to code, data, and documentation, are the proprietary work of [Kayalvizhi Kamatchi Selvaraj]. No part of this repository may be reproduced, distributed, or used in any form without prior written permission from the author.

Permission is explicitly required for:
- Use in academic or commercial projects.
- Distribution or modification of the contents of this repository.

To request permission, contact **[https://www.linkedin.com/in/kayalvizhiks/]**.
