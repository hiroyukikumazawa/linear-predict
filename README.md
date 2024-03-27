# Linear Regression Model Predictor

This project is a simple yet powerful implementation of a linear regression model served through a FastAPI application. It's designed to predict outcomes based on a pre-trained TensorFlow model. The API is intuitive and documented, making it accessible for developers and machine learning enthusiasts.

## Features

- **FastAPI Framework**: Utilizes FastAPI for efficient and easy-to-use API endpoints.
- **TensorFlow Integration**: Leverages a TensorFlow model for making predictions.
- **Auto-Generated Documentation**: Includes Swagger UI documentation generated automatically by FastAPI.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or above
- pip for package installation

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/hiroyukikumazawa/linear-predict.git
   ```
2. Navigate to the project directory:
   ```bash
   cd linear-predict
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To start the FastAPI server, run the following command:

```bash
uvicorn app:app --reload
```

This command will start the development server with live reloading. The API documentation will be available at `http://127.0.0.1:8000/docs`.

### Making Predictions

To make predictions, use the `/predict/` endpoint. You can do this through the Swagger UI or by sending a POST request with a JSON body containing the values for prediction:

```json
{
  "values": [0.1, 0.5, 0.9]
}
```

### Training the Model

If you wish to retrain the model with your data, run:

```bash
python ./train/train.py
```

This script generates synthetic training data, trains the linear regression model, and saves it for the application to use.

## Contributing

We welcome contributions! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Hiroyuki Kumazawa - hiroyukikumazawa.jp@gmail.com

Project Link: [https://github.com/hiroyukikumazawa/linear-predict](https://github.com/hiroyukikumazawa/linear-predict)

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow](https://www.tensorflow.org/)
