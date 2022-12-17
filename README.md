# wine-api

<img src="assets/fastapi-logo.png" alt="fastapi-logo" width="300"/><img src="assets/Scikit_learn_logo_small.svg.png" alt="sklearn-logo" width="300"/>

## Overview

This is an AI-based project to determine which wine composition is the best.
The dataset used for training the model is available in the [data](data/) directory of this git repository.

This project use [FastAPI](https://fastapi.tiangolo.com/) framework to create the API which interact with the [scikit-learn](https://scikit-learn.org/stable/) AI-model

## Installation

To install all the requirements, use the package manager [pip](https://pip.pypa.io/en/stable/) and type this in your terminal.

```bash
pip install -r /path/to/requirements.txt
```

## File structure

```
|-- app
|   |-- classes
|   |   |-- __pycache__
|   |   |   `-- models.cpython-38.pyc
|   |   `-- models.py
|   |-- core
|   |   |-- __pycache__
|   |   |   `-- config.cpython-38.pyc
|   |   `-- config.py
|   |-- models
|   |   |-- metrics.json
|   |   `-- model.pkl
|   |-- __pycache__
|   |   `-- main.cpython-38.pyc
|   |-- routes
|   |   |-- __pycache__
|   |   |   |-- model.cpython-38.pyc
|   |   |   `-- predict.cpython-38.pyc
|   |   |-- model.py
|   |   `-- predict.py
|   |-- scripts
|   |   |-- __pycache__
|   |   |   |-- model_tools.cpython-38.pyc
|   |   |   `-- predict_tools.cpython-38.pyc
|   |   |-- model_tools.py
|   |   `-- predict_tools.py
|   `-- main.py
|-- assets
|   |-- fastapi-logo.png
|   |-- redoc.png
|   |-- Scikit_learn_logo_small.svg.png
|   `-- swagger.png
|-- data
|   `-- Wines.csv
|-- data-exploration
|   `-- data_explo.ipynb
|-- Dockerfile
|-- README.md
`-- requirements.txt
```

## Usage

First of all, you have to launch the app :

```bash
uvicorn app.main:app --reload --port 8000
```

If you want to change the port, you can change the 8000 value by the port you want.

This command should create the local server on your computer with the app running on it.

Then, to visualize and test all the endpoints and the API routes available, go to [http://localhost:8000/docs](http://localhost:8000/docs) and you should see the swagger page like this :

![swagger](./assets/swagger.png)

You can also see a more detailled documentation at [http://localhost:8000/redoc](http://localhost:8000/redoc) like :

![swagger](./assets/redoc.png)
