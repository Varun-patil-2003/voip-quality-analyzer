# VoIP Quality Analyzer with Machine Learning 📞🎧

An intelligent, data-driven network analysis tool that leverages Machine Learning to predict and analyze Voice over IP (VoIP) call quality. By monitoring and evaluating critical network performance metrics—such as jitter, latency, and packet loss—this application classifies call quality, calculates an estimated Mean Opinion Score (MOS), and provides actionable network optimization insights.

---

## Key Features

- **Predictive Quality Analysis:** Uses trained Machine Learning classification models to categorize call quality (e.g., Excellent, Good, Fair, Poor).
- **Network Telemetry Integration:** Captures or simulates key performance indicators (KPIs) critical to VoIP environments.
- **Detailed Metrics Evaluation:** Calculates delay variation (jitter), round-trip time (latency), packet loss percentage, and translates them into an estimated MOS.
- **Interactive Visualization:** Displays evaluation trends, dataset distributions, and performance graphs for intuitive troubleshooting.

---

## Technology Stack

- **Language:** Python 3.8+
- **Machine Learning & Data Science:** Scikit-Learn, Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Development Tooling:** Git, Jupyter Notebooks (if applicable)

---

## Project Workflow

The application operates through a structured pipeline:
[ Network Traffic / Data Input ]
│
▼
[ Feature Extraction ]
(Latency, Jitter, Packet Loss, MOS)
│
▼
[ Machine Learning Model ]
(e.g., Random Forest / Decision Trees)
│
▼
[ Quality Classification Output ]
(Excellent, Good, Fair, Poor)


1. **Data Preprocessing & Cleaning:** Normalizes the incoming dataset, handles missing values, and processes network telemetry variables.
2. **Feature Engineering:** Computes statistical values from raw network packet parameters to generate predictive inputs.
3. **Model Training & Evaluation:** Trains supervised learning algorithms and validates accuracy using classification reports, confusion matrices, and ROC curves.
4. **Inference:** Accepts live or simulated test metrics to output instant quality ratings.

---

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites
Make sure you have Python 3.8 or higher installed. You can check your version by running:
``` python --version ```

### Installation Steps
#### Clone the Repository:
``` git clone [https://github.com/Varun-patil-2003/voip-quality-analyzer.git](https://github.com/Varun-patil-2003/voip-quality-analyzer.git) ```
``` cd voip-quality-analyzer ```

#### Create a Virtual Environment (Recommended):
On Windows:
``` python -m venv venv ``` 
``` venv\Scripts\activate ```

On macOS/Linux:
``` python3 -m venv venv ```
``` source venv/bin/activate ```

#### Install Dependencies:
Ensure you have all the required libraries installed:
``` pip install -r requirements.txt ```

### How to Run
1. Training the Model
To train the machine learning classifier on your dataset and save the serialized model file:
```
python train.py
```
(Note: If your training pipeline is in a notebook, open it using jupyter notebook or VS Code and run train.ipynb)

2. Running Predictions
To run the analyzer tool and evaluate test data:
``` python main.py ```

### Future Enhancements
Real-Time Packet Capture: Integrate direct network packet sniffing using libraries like Scapy or PyShark to analyze active network interfaces.

Deep Learning Integration: Explore LSTM (Long Short-Term Memory) networks for time-series forecasting of packet degradation.

Interactive Web Interface: Build a frontend dashboard using Streamlit or Flask to monitor ongoing VoIP stats visually.

### Contributing
Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

    Fork the Project

    Create your Feature Branch (git checkout -b feature/AmazingFeature)

    Commit your Changes (git commit -m 'Add some AmazingFeature')

    Push to the Branch (git push origin feature/AmazingFeature)

    Open a Pull Request

### License
This project is licensed under the MIT License. See the LICENSE file for more details.


***

### Quick Tips Before Committing:
1. **Requirements:** Run the command `pip freeze > requirements.txt` from inside your virtual environment to generate your dependency list before pushing.
2. **Adjust Commands:** If your entry files have different names (e.g., `model.py` inst

---

[Click here to run application](https://varun-patil-2003-voip-quality-analyzer-appstreamlit-ui-hhms5h.streamlit.app/)