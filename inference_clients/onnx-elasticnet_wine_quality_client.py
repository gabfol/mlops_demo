#!pip install open-inference-openapi
from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import os


# If running from workbench use /tmp/jwt. Otherwise provide your CDP_TOKEN
API_KEY = json.load(open("/tmp/jwt"))["access_token"]


BASE_URL = 'https://caii-inference.emerging.dp5i-5vkq.cloudera.site/namespaces/serving-default/endpoints/elasticnetwine2141onnx'
MODEL_NAME = 'a00d-9ysj-zlk6-hu3t'
headers = {
	'Authorization': 'Bearer ' + API_KEY,
	'Content-Type': 'application/json'
}

httpx_client = httpx.Client(headers=headers)
client = OpenInferenceClient(base_url=BASE_URL, httpx_client=httpx_client)

# Check that the server is live, and it has the model loaded
client.check_server_readiness()
metadata = client.read_model_metadata(MODEL_NAME)
metadata_str = json.dumps(json.loads(metadata.json()), indent=2)
print(metadata_str)


# Load JSON file (list of dicts)
with open("sample_inputs/X_test_first_record.json", "r") as f:
    records = json.load(f)  # e.g. [ {dict}, {dict}, ... ]

# Extract columns in order from the first record's keys
columns = list(records[0].keys())

# Build data values as list of lists in column order
data_values = [
    [record[col] for col in columns] 
    for record in records
]

# Prepare inputs payload
inputs_test = [
    {
        "name": "input",
        "datatype": "FP32",
        "shape": [len(data_values), len(columns)],
        "data": data_values
    }
]

print(json.dumps(inputs_test, indent=2))


# Make an inference request
pred = client.model_infer(
	MODEL_NAME,
	request=InferenceRequest(
		inputs=inputs_test
	)
)

json_resp_str = json.dumps(json.loads(pred.json()), indent=2)
print(json_resp_str)