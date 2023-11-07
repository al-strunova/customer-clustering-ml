# Customer Clustering

Identify distinct customer segments using a machine learning clustering model.

## Features
- Developed with Scikit-learn and FastAPI.
- Utilizes the K-Means clustering algorithm.
- Docker-ready for easy deployment.

## Project Structure
- **notebooks**: Contains a Jupyter notebook used for both analysis and model experimentation.
  - `Customer_Segmentation_EDA.ipynb`: This notebook starts with an Exploratory Data Analysis (EDA) to gain insights into the customer data and then explores various clustering techniques, including K-Means, Hierarchical Clustering, and DBSCAN. The notebook details the performance of these methods, with K-Means slightly outperforming Hierarchical Clustering. DBSCAN did not perform well due to the data's characteristics. K-Means was selected for the final model due to its scalability and ability to handle new, unseen data, which is not feasible with Hierarchical Clustering.
- **data**: Directory containing the datasets used for model training and analysis.
  > **Note**: Be mindful of privacy and licensing when storing and sharing data. The included dataset is for demonstration purposes only.
- **src**: Contains the core scripts for data preprocessing, model training, and inference.
- **models**: Stores the serialized versions of the trained clustering model and any associated preprocessing objects.
- **templates**: Consists of XML files tasked with rendering the user interface.

## Prerequisites
- Python 3.9+
- Docker (optional, for containerization)

## Getting Started

### Local Setup
1. **Clone**:
   ```sh
   git clone https://github.com/your-username/customer-clustering-ml.git
   
2. **Environment and Dependencies**:
   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Run FastAPI**:
   ```sh
   cd src
   uvicorn app:app --reload

### Docker Deployment
1. **Build Image**:
   ```sh
   docker build -t customer-clustering-ml .
2. **Run Container**:
   ```sh
   docker run -p 8000:8000 customer-clustering-ml

Open http://localhost:8000/ to access the UI.

## Endpoints
- '/' : main page to interact with ui
- '/predict' : Get prediction on loan repayment

## Contributing
PRs are welcome. For major changes, open an issue first.

## License
MIT License
