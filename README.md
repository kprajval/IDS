# IDS

<b><i>Intrusion Detection System (IDS) Using Machine Learning</i></b>
This project focuses on building an advanced Intrusion Detection System (IDS) using machine learning techniques to detect and classify network attacks effectively.

<b>🔍 Model Overview:<b>
An ensemble model was created by integrating three powerful classifiers:
  - Random Forest
  - Decision Tree
  - XGBoost

For every prediction, input data is passed through all three models. The final classification is determined using a majority voting mechanism, enhancing robustness and improving accuracy across various types of network traffic.

<b>📊 Dataset:</b>
The dataset was sourced from Kaggle, containing around 1 million records. It includes a diverse set of network traffic samples, ensuring comprehensive training and high generalization capability.

<b>🧪 Attack Simulation:<b>
To validate the system, a Python script named attack_sim.py is included. This script launches sample simulated attacks using single attack vectors (e.g., DoS, probe, R2L, U2R) to test the model's classification performance in controlled conditions. It helps verify if the IDS can effectively detect and classify various types of intrusions in real-time scenarios.

<b>✅ Outcome:<b>
This approach delivers efficient and accurate results, demonstrating the practicality of ensemble-based detection in real-world cybersecurity applications.
