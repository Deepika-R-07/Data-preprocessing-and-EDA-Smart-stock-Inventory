📦 Smart Stock Inventory — Data Preprocessing, EDA & Forecasting

An end-to-end **Smart Stock Inventory Management and Analysis** project that uses data preprocessing, exploratory data analysis (EDA), demand forecasting, and inventory analytics to support better stock management and data-driven decision-making.

 🚀 Live Demo

🔗 Streamlit Dashboard:
https://data-preprocessing-and-eda-smart-stock-inventory-gu52amkugpjzz.streamlit.app/

🔗GitHub Repository:
https://github.com/Deepika-R-07/Data-preprocessing-and-EDA-Smart-stock-Inventory



📌 Project Overview

Inventory management is an important part of retail and supply-chain operations. Poor inventory planning can lead to:

* 📉 Stockouts
* 📦 Overstocking
* 💰 Increased holding costs
* ⚠️ Poor demand planning
* 🔄 Inefficient replenishment decisions

This project analyzes historical inventory and sales data to identify patterns and trends and uses forecasting techniques to estimate future demand.

The project follows a complete data analytics pipeline:


Raw Inventory Data
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
        ↓
Demand Forecasting
        ↓
Inventory Analysis
        ↓
Interactive Streamlit Dashboard
        ↓
Business Insights & Decisions




🎯 Objectives

The main objectives of this project are:

1. Clean and preprocess raw inventory data.
2. Handle missing values, duplicates, and inconsistent data.
3. Perform exploratory data analysis.
4. Identify sales and inventory trends.
5. Analyze product-level and time-based patterns.
6. Forecast future product demand.
7. Generate useful inventory insights.
8. Visualize results through an interactive dashboard.
9. Support data-driven inventory management decisions.


🛠️ Technologies Used

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| 🐍 Python             | Core programming language           |
| 🐼 Pandas             | Data manipulation and preprocessing |
| 🔢 NumPy              | Numerical operations                |
| 📊 Matplotlib         | Data visualization                  |
| 📈 Seaborn            | Statistical visualization           |
| 🤖 Scikit-learn       | Machine learning utilities          |
| 🔮 Prophet            | Time-series forecasting             |
| 🧠 TensorFlow / Keras | LSTM forecasting                    |
| 📓 Jupyter Notebook   | Data analysis                       |
| 🎨 Streamlit          | Interactive dashboard               |
| 🐙 Git & GitHub       | Version control                     |



🧹 1. Data Preprocessing

The raw inventory dataset is first cleaned and transformed before performing analysis and forecasting.

Major preprocessing steps

* Handling missing values
* Removing duplicate records
* Detecting inconsistent values
* Converting data types
* Date/time preprocessing
* Handling outliers
* Feature transformation
* Preparing data for forecasting

The goal is to produce a clean and reliable dataset for downstream analysis.



📊 2. Exploratory Data Analysis

EDA is performed to understand the characteristics of the inventory and sales data.

Analysis includes

* Product-wise sales analysis
* Inventory level analysis
* Sales trends over time
* Product demand distribution
* Category-wise analysis
* Correlation analysis
* Outlier detection
* Time-based patterns
* Inventory and sales relationships

📈 Visualizations

The project uses different charts to make the analysis easier to understand, including:

* Bar charts
* Line charts
* Histograms
* Box plots
* Heatmaps
* Time-series plots
* Product/category comparisons

EDA helps identify important patterns that can be used during forecasting and inventory planning.



🔮 3. Demand Forecasting

The project includes forecasting techniques to estimate future demand.

Forecasting helps answer questions such as:

> **How much inventory may be required in the future?**

The forecasting component can be used to identify future demand patterns and support inventory planning.

 Forecasting workflow:


Historical Sales Data
        ↓
Time-Series Preparation
        ↓
Feature Preparation
        ↓
Forecasting Model
        ↓
Future Demand Prediction
        ↓
Inventory Planning


Forecast results are generated and used by the inventory dashboard for further analysis.



📦 4. Smart Inventory Analysis

The forecasting results can be used to support inventory decisions such as:

* Identifying products with high demand
* Detecting low-stock situations
* Understanding demand variability
* Planning future inventory requirements
* Supporting replenishment decisions
* Reducing the possibility of stockouts
* Avoiding unnecessary overstocking

The overall goal is to balance **product availability and inventory cost**.



🎨 5. Interactive Streamlit Dashboard

The project includes an interactive dashboard built using **Streamlit**.

Dashboard capabilities

📊 Data Overview

Provides an overview of the inventory dataset and important statistics.

📈 EDA & Visualization

Displays trends, distributions, comparisons, and other analytical visualizations.

🔮 Forecasting

Displays forecasted demand and forecasting results.

📦Inventory Insights

Helps identify important inventory-related patterns and potential stock issues.

📋 Reports

Provides summarized information that can be used for decision-making.



🌐 Live Application

You can access the deployed application here:

👉 https://data-preprocessing-and-eda-smart-stock-inventory-gu52amkugpjzz.streamlit.app/



📂 Project Structure


Data-preprocessing-and-EDA-Smart-stock-Inventory/
│
├── Data Preprocessing and EDA/
│   └── Data preprocessing and EDA files
│
├── Forecasting model/
│   └── Forecasting model files
│
├── Smart Inventory Dashboard/
│   └── Dashboard application files
│
├── Smart Inventory Dashboard & Reporting/
│   └── Dashboard and reporting files
│
├── LICENSE
│
└── README.md




⚙️ Installation

### 1. Clone the repository

bash
git clone https://github.com/Deepika-R-07/Data-preprocessing-and-EDA-Smart-stock-Inventory.git

2. Navigate to the project directory

bash
cd Data-preprocessing-and-EDA-Smart-stock-Inventory

 3. Create a virtual environment

bash
python -m venv venv


4. Activate the environment

 Windows

bash
venv\Scripts\activate


Linux / macOS

bash
source venv/bin/activate

5. Install required packages

bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit prophet tensorflow




▶️ Running the Dashboard

Navigate to the folder containing the Streamlit application and run:

bash
streamlit run app.py


The dashboard will open in your browser.

📊 Project Workflow
                 ┌───────────────────┐
                 │   Raw Inventory   │
                 │       Data        │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ Data Preprocessing│
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │       EDA         │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ Feature Engineering│
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │    Forecasting    │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ Inventory Analysis│
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ Streamlit Dashboard│
                 └───────────────────┘



💡 Key Benefits

📈 Better Demand Planning

Forecasting provides an estimate of future demand.

📦 Improved Inventory Management

Inventory patterns can be analyzed to identify potential stock issues.

💰 Reduced Inventory Costs

Better planning can help reduce unnecessary inventory and holding costs.

⚡ Faster Decision Making

The interactive dashboard provides important information in an easy-to-understand format.

📊 Data-Driven Insights

EDA and visualization help transform raw inventory data into useful business insights.



🔮 Future Enhancements

The project can be extended with:

* Real-time inventory tracking
* Automated low-stock alerts
* Email/SMS notifications
* Advanced forecasting models
* XGBoost forecasting
* Improved LSTM architecture
* Automated reorder recommendations
* EOQ calculation
* Safety Stock calculation
* Reorder Point calculation
* ABC inventory classification
* Cloud database integration
* User authentication
* Role-based dashboards



🎓 Learning Outcomes

Through this project, the following concepts were explored:

* Data cleaning
* Data preprocessing
* Exploratory Data Analysis
* Data visualization
* Feature engineering
* Time-series forecasting
* Machine learning
* Inventory analytics
* Streamlit application development
* Git and GitHub
* Cloud deployment


👩‍💻 Author

**Deepika Rajagopal**
B.Tech Information Technology

GitHub:
https://github.com/Deepika-R-07


# 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

