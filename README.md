# Settyl API Assignment

## Overview

Settyl- Container Event is a web API built with FastAPI that provides predictive analysis for internal status based on external status inputs. It uses a pre-trained TensorFlow model to make predictions.

## Base URL

The base URL for accessing the Settyl API is `http://localhost:8000`.

## Authentication

This API does not require authentication.

## Endpoints

### Root

- **URL**: `/`
- **Method**: GET
- **Description**: Returns a welcome message indicating successful connection to the API.

### Prediction

- **URL**: `/predict/`
- **Method**: POST
- **Description**: Makes a prediction for the internal status based on the provided external status.
- **Request Body**:
  ```json
  {
    "externalStatus": "string"
  }
  ```
- **Response Body**:
  ```json
  {
    "predicted_internal_status": integer
  }
  ```

## Sample Usage

### Request

```http
POST /predict/ HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "externalStatus": "Your input data here"
}
```

### Response

```json
{
  "predicted_internal_status": 8
}
```

## Error Handling

- **Status Code 400**: Bad Request. Indicates that the request body is missing or malformed.
- **Status Code 500**: Internal Server Error. Indicates an unexpected error occurred on the server.

## Dependencies

- **FastAPI**: Web framework for building APIs.
- **Pydantic**: Data validation and parsing library.
- **NumPy**: Numerical computing library for handling array operations.
- **TensorFlow**: Machine learning framework for building and deploying models.
- **Logging**: Standard Python logging module for logging debug and error messages.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
