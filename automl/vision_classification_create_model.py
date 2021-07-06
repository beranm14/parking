import os
from google.cloud import automl

project_id = os.getenv("PROJECT_ID")
dataset_id = "ICN7022277851151859712"
display_name = os.getenv("DISPLAY_NAME")

client = automl.AutoMlClient()

# A resource that represents Google Cloud Platform location.
project_location = f"projects/{project_id}/locations/us-central1"
# Leave model unset to use the default base model provided by Google
# train_budget_milli_node_hours: The actual train_cost will be equal or
# less than this value.
# https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#imageclassificationmodelmetadata
metadata = automl.ImageClassificationModelMetadata(
    train_budget_milli_node_hours=24000
)
model = automl.Model(
    display_name=display_name,
    dataset_id=dataset_id,
    image_classification_model_metadata=metadata,
)

# Create a model with the model metadata in the region.
response = client.create_model(parent=project_location, model=model)

print("Training operation name: {}".format(response.operation.name))
print("Training started...")
