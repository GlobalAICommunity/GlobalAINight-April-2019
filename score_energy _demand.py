import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame(data=[{
              "timestamp": "2012-01-01 00:00:00",
              "precip": 0.1,
              "input3": 23.4
            }])
output_sample = np.array([12.3])

def init():
    global model
    model_path = Model.get_model_path('<<model_name>>')
    model = joblib.load(model_path)

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result": result.tolist()})