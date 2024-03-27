from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse

from models import ValueList
from predict import predict

# Initialize the FastAPI app
app = FastAPI(
    title="Linear Regression Model Predictor",
    description="A simple API to make predictions using a pre-trained linear regression model.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    """
    Redirect to API documentation.

    This endpoint redirects users to the automatically generated Swagger UI documentation for the API,
    making it easier to interact with and explore available API functionalities.
    """
    return RedirectResponse(url="/docs")


@app.post("/predict/", summary="Predict values using a pre-trained model")
def process_values(data: ValueList):
    """
    Process values and predict outcomes.

    Receives a list of float values as input, uses a pre-trained linear regression model to predict the outcomes,
    and returns the predictions as a list of floats.

    Parameters:
    - data: A list of floats provided by the client in the body of the POST request, adhering to the ValueList model.

    Returns:
    - A JSON object containing the list of predicted values.
    """
    try:
        print(f"Received values for prediction: {data.values}")
        # Reformatting the input data to match the expected shape for prediction
        values = [[item] for item in data.values]
        result = predict(values)
        return {"result": result.tolist()}
    except Exception as e:
        # Handling unexpected errors during prediction
        raise HTTPException(status_code=500, detail=str(e))
