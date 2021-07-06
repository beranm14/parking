import os

from google.cloud import automl

# TODO(developer): Uncomment and set the following variables
project_id = os.getenv("PROJECT_ID")
dataset_id = "ICN7022277851151859712"
display_name = os.getenv("DISPLAY_NAME")
model_id = "ICN4565073322479452160"

client = automl.AutoMlClient()
# Get the full path of the model.
model_full_id = client.model_path(project_id, "us-central1", model_id)

print("List of model evaluations:")
for evaluation in client.list_model_evaluations(parent=model_full_id, filter=""):
    print("Model evaluation name: {}".format(evaluation.name))
    print("Model annotation spec id: {}".format(evaluation.annotation_spec_id))
    print("Create Time: {}".format(evaluation.create_time))
    print("Evaluation example count: {}".format(evaluation.evaluated_example_count))
    print(
        "Classification model evaluation metrics: {}".format(
            evaluation.classification_evaluation_metrics
        )
    )