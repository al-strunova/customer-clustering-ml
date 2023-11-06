import io
import os

from fastapi.templating import Jinja2Templates
import pandas as pd
from pydantic import BaseModel
import logging
from src.clustering_predictor import PredictClustering
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from fastapi import Request

# Setting up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Get the absolute path to the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct absolute path to the templates directory
templates_directory = os.path.join(script_directory, '..', 'templates')
templates = Jinja2Templates(directory=templates_directory)

app = FastAPI()

# Construct the absolute path to the data directory
data_directory = os.path.join(script_directory, '..', 'data')
app.mount("/data", StaticFiles(directory=data_directory), name="data")

# Create an instance of the model
model = PredictClustering()


@app.get("/")
def ui_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict_csv(file: UploadFile):
    try:
        # Read the uploaded CSV file and convert it to a DataFrame
        csv_content = await file.read()
        input_df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))

        # Perform predictions using your model
        predictions = model.predict(input_df)

        # Convert the DataFrame to a CSV format in memory
        output = io.StringIO()
        predictions.to_csv(output, index=False)
        output.seek(0)

        # Create a temporary CSV file for download
        tmpfile_path = None
        with NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
            tmpfile.write(output.getvalue().encode('utf-8'))
            tmpfile_path = tmpfile.name

        # Return the temporary file for download outside the with block
        with open(tmpfile_path, "rb") as f:
            return StreamingResponse(
                iter([f.read()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=predictions.csv"},
            )

    except Exception as e:
        logging.error(f"An error occurred while making predictions: {e}")
        return {"detail": "Internal server error"}
