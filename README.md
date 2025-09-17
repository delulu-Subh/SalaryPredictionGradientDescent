# Salary Prediction using Linear Regression with Gradient Descent 💰

Predict employee salaries based on years of experience using single-feature linear regression trained with gradient descent. This project includes hyperparameter tuning, iterative visualization, and performance evaluation with R² score.

Project Overview

This project demonstrates:

Single-feature linear regression to predict salaries.

Gradient descent optimization for slope (theta1) and intercept (theta0).

Visualization of hypothesis function updating after each epoch.

Model evaluation using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Repository Structure
SalaryPrediction/
│
├─ salary_dataset.csv             # Dataset with 'YearsExperience' and 'Salary'
├─ SalaryGD.ipynb                 # Jupyter notebook implementing gradient descent
├─ README.md                      # Project documentation
├─ requirements.txt               # Python dependencies
└─ epoch_plot.png                 # Visualization of hypothesis function vs data points

Features

Single Feature Regression: Predict salaries from years of experience.

Gradient Descent Training: Learn optimal parameters iteratively.

Hyperparameter Tuning: Adjust learning rate and epochs for best results.

Real-Time Visualization: Watch the hypothesis function adapt to training data.

Evaluation Metrics: Includes MAE, MSE, RMSE, and R².

Example Visualization

Installation
pip install -r requirements.txt

Usage

Open SalaryGD.ipynb in Jupyter Notebook.

Run the notebook to:

Load and preprocess the salary dataset.

Train the gradient descent model on training data.

Visualize the hypothesis function at each epoch.

Evaluate model performance on test data.

Check epoch_plot.png for the final visualization of the fitted line.


Sample Output
Hypothesis Function: h(x) = 25792.0 + 9449.9 x
R² Score: 0.97
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/6ad2d9aa-3277-4566-8984-1e1e806ec91e" />
Hypothesis Function: h(x) = 3062.440182954563  + 13050.662814163326 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/724016dc-bd71-40ca-be24-8f02a535b78f" />
Hypothesis Function: h(x) = 3110.531837108589  + 13044.536277106688 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/c58480ad-48b9-4d2b-b88b-94de324717ad" />
Hypothesis Function: h(x) = 3158.4524753871897  + 13037.953422622946 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/8b5621ef-2931-4f58-9ff5-be7c4a2a7405" />
Hypothesis Function: h(x) = 3206.2268900056406  + 13031.070101661826 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/0ba0540d-faed-4838-ba41-1ffbe0187646" />
Hypothesis Function: h(x) = 3253.8717417965236  + 13023.990636181297 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/8dfef0a3-3e42-4798-85ee-6e3901d901c0" />
Hypothesis Function: h(x) = 3301.398248031721  + 13016.78485285394 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/f9757d88-966f-4134-bab3-655963cb9a98" />
Hypothesis Function: h(x) = 3348.8139817445417  + 13009.499486024535 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/94f507bd-0b62-4a04-a47b-2fa434c3c012" />
Hypothesis Function: h(x) = 3396.124076259022  + 13002.165811247876 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/84b2e86d-a3e9-4987-8c13-dee431dc3c90" />
Hypothesis Function: h(x) = 3443.3320315441433  + 12994.804755445397 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/d5bf8036-10de-4891-b2d4-1e0376e7fc3f" />
Hypothesis Function: h(x) = 3490.4402540157557  + 12987.430317822362 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/3efc23c6-927f-4145-b585-f3c67db53e7a" />
Hypothesis Function: h(x) = 3537.4504178990837  + 12980.0518599492 x
<img width="598" height="453" alt="download" src="https://github.com/user-attachments/assets/8d390ecf-8147-41d6-baea-d8f44b2f4273" />


The above output shows the learned parameters and the R² score of the trained model.

Performance Metrics

Training complete.
R² Score: {r2_score:.4f}
0.8975563610527842

(Actual values will depend on dataset and hyperparameters.)

Next Steps / Improvements

Extend to multiple features for more accurate salary prediction.

Implement learning rate schedules for faster convergence.

Compare with scikit-learn’s LinearRegression for benchmarking.
