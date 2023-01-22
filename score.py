import os
import json
import time
import logging
import datetime
from pathlib import Path
import onnxruntime as rt
import numpy as np
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

session, transform, classes, input_name, model_stamp = None, None, None, None, None
logger = logging.getLogger()

# adding additional logging to app insights
if 'AML_APP_INSIGHTS_KEY' in os.environ:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    logger.addHandler(AzureLogHandler(
        connection_string='InstrumentationKey=' + os.environ['AML_APP_INSIGHTS_KEY'])
    )


def init():
    global session, transform, classes, logger, model_stamp
    logger.info('initializing...')
    logger.info(os.environ)
    if 'AZUREML_MODEL_DIR' in os.environ:
        root_dir = Path(os.environ['AZUREML_MODEL_DIR']).resolve() / 'model'
    else:
        root_dir = Path('outputs/model').absolute().resolve()

    logger.info(f'using model path {root_dir}')    
    meta_file = root_dir / 'model.json'
    model_file = root_dir / 'model.onnx'
    logger.info(f'metadata path: {meta_file}')
    logger.info(f'model path: {model_file}')

    with open(meta_file, 'r') as f:
        model_meta = json.load(f)
    logger.info(f'metadata load complete: {model_meta}')

    transform = model_meta['transforms']
    model_stamp = model_meta['timestamp']

    session = rt.InferenceSession(str(model_file),
                    providers=['CUDAExecutionProvider'])
    
    logger.info(f'init complete!')

def get(val, col: str):
    xform = transform[col]
    if isinstance(xform, dict):
        return xform[val.lower()]
    elif isinstance(xform, list):
        return (val - xform[0]) / (xform[1] - xform[0])
    else:
        return 1 if xform == val else 0

@input_schema("programmer", StandardPythonParameterType({
    'age': StandardPythonParameterType(34.0),
    'location': StandardPythonParameterType('South America'),
    'orgsz': StandardPythonParameterType(751.0),
    'style': StandardPythonParameterType('spaces'),
    'yoe': StandardPythonParameterType(19.0),
    'projects': StandardPythonParameterType(26.0)
}))
@output_schema(StandardPythonParameterType({
  'time': StandardPythonParameterType(0.060392),
  'prediction': StandardPythonParameterType("yes"),
  'scores': StandardPythonParameterType({
    'yes': StandardPythonParameterType(1.0),
    'no': StandardPythonParameterType(0.0),
  }),
  'timestamp': StandardPythonParameterType(datetime.datetime.now().isoformat()),
  'model_update': StandardPythonParameterType(datetime.datetime.now().isoformat()),
  'message': StandardPythonParameterType("Success!")
}))
def run(programmer):
    global session, transform, classes, logger, model_stamp

    logger.info(f'request: {json.dumps(programmer)}')
    prev_time = time.time()

    # process data
    try:
        v = [get(programmer[x], x) for x in transform.keys() if x in programmer]

        data = {
            'location': np.array([v[0]]).astype('int64'),
            'style': np.array([v[1]]).astype('int64'),
            'numerics': np.array([v[2:]]).astype('float32')
        }
        logger.info(f'location: {data["location"].tolist()}, style: {data["style"].tolist()}, numerics: {data["numerics"].tolist()}')
        pred_onnx = session.run(None, data)
        probs = pred_onnx[0][0][0]

        payload = {
            'time': float(0),
            'prediction': 'yes' if probs > 0.5 else 'no',
            'scores': {"yes": float(probs), "no": float(1-probs)},
            'timestamp': datetime.datetime.now().isoformat(),
            'model_update': model_stamp,
            'message': 'Success!'
        }

    except Exception as e:
        logger.error(e)
        payload = {
            'time': float(0),
            'prediction': "none",
            'scores': {"yes": 0, "no": 0},
            'timestamp': datetime.datetime.now().isoformat(),
            'model_update': model_stamp,
            'message': f'{e}'
        }

    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    payload['time'] = float(inference_time.total_seconds())

    logger.info(f'payload: {json.dumps(payload)}')

    return payload


        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.INFO)
    init()
    one = {
        'age': 34.0,
        'location': 'South America',
        'orgsz': 751.0,
        'style': 'spaces',
        'yoe': 19.0,
        'projects': 26.0
    }
    resp1 = run(programmer=one)
    print(json.dumps(resp1, indent=2))

    two = {
        'age': 36.0,
        'location': 'Oceania',
        'orgsz': 620.0,
        'style': 'spaces',
        'yoe': 17.0,
        'projects': 13.0
    }
    resp2 = run(programmer=two)
    print(json.dumps(resp2, indent=2))
    