import os
import sys
from google.cloud import automl

# TODO(developer): Uncomment and set the following variables
project_id = os.getenv("PROJECT_ID")
dataset_id = "ICN7022277851151859712"
display_name = os.getenv("DISPLAY_NAME")
model_id = "ICN4565073322479452160"
# file_path = "testing_photos/210705_161002.jpg"
# file_path = "testing_photos/210705_161002.jpg"
file_path = "testing_photos/210705_140001.jpg"

if len(sys.argv) > 1:
    file_path = sys.argv[1]

prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = automl.AutoMlClient.model_path(project_id, "us-central1", model_id)

# Read the file.
with open(file_path, "rb") as content_file:
    content = content_file.read()

image = automl.Image(image_bytes=content)
payload = automl.ExamplePayload(image=image)

# params is additional domain-specific parameters.
# score_threshold is used to filter the result
# https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#predictrequest
params = {"score_threshold": "0.4"}

request = automl.PredictRequest(name=model_full_id, payload=payload, params=params)
response = prediction_client.predict(request=request)

print("Prediction results:")
for result in response.payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))
