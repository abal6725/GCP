
###Create a Dataset
project_id = 'global-brook-217321'
compute_region = 'us-central1'
dataset_name = 'TwitterSentiment'
multilabel = False


from google.cloud import automl_v1beta1 as automl

client = automl.AutoMlClient()

# A resource that represents Google Cloud Platform location.
project_location = client.location_path(project_id, compute_region)

# Classification type is assigned based on multilabel value.
classification_type = "MULTICLASS"
if multilabel:
    classification_type = "MULTILABEL"

# Specify the text classification type for the dataset.
dataset_metadata = {"classification_type": classification_type}

# Set dataset name and metadata.
my_dataset = {
    "display_name": dataset_name,
    "text_classification_dataset_metadata": dataset_metadata,
}

# Create a dataset with the dataset metadata in the region.
dataset = client.create_dataset(project_location, my_dataset)

# Display the dataset information.
print("Dataset name: {}".format(dataset.name))
print("Dataset id: {}".format(dataset.name.split("/")[-1]))
print("Dataset display name: {}".format(dataset.display_name))
print("Text classification dataset metadata:")
print("\t{}".format(dataset.text_classification_dataset_metadata))
print("Dataset example count: {}".format(dataset.example_count))
print("Dataset create time:")
print("\tseconds: {}".format(dataset.create_time.seconds))
print("\tnanos: {}".format(dataset.create_time.nanos))

###Import Training Items into the Dataset
dataset_id = 'TCN5191523838201987133'
path = 'gs://global-brook-217321-lcm/'

# Get the full path of the dataset.
dataset_full_id = client.dataset_path(
    project_id, compute_region, dataset_id
)

# Get the multiple Google Cloud Storage URIs.
input_uris = path.split(",")
input_config = {"gcs_source": {"input_uris": input_uris}}


print("Processing import...")
# synchronous check of operation status.
print("Data imported. {}".format(response.result()))

###Create(train) the model
model_name = 'TwitterSentiment'

# Set model name and model metadata for the dataset.
my_model = {
    "display_name": model_name,
    "dataset_id": dataset_id,
    "text_classification_model_metadata": {},
}

# Create a model with the model metadata in the region.
response = client.create_model(project_location, my_model)
print("Training operation name: {}".format(response.operation.name))
print("Training started...")


### Evaluate the Model

# Get the full path of the model.
model_full_id = client.model_path(project_id, compute_region, model_id)

# List all the model evaluations in the model by applying filter.
response = client.list_model_evaluations(model_full_id, filter_)

# Iterate through the results.
for element in response:
    # There is evaluation for each class in a model and for overall model.
    # Get only the evaluation of overall model.
    if not element.annotation_spec_id:
        model_evaluation_id = element.name.split("/")[-1]

# Resource name for the model evaluation.
model_evaluation_full_id = client.model_evaluation_path(
    project_id, compute_region, model_id, model_evaluation_id
)

# Get a model evaluation.
model_evaluation = client.get_model_evaluation(model_evaluation_full_id)

class_metrics = model_evaluation.classification_evaluation_metrics
confidence_metrics_entries = class_metrics.confidence_metrics_entry

# Showing model score based on threshold of 0.5
for confidence_metrics_entry in confidence_metrics_entries:
    if confidence_metrics_entry.confidence_threshold == 0.5:
        print("Precision and recall are based on a score threshold of 0.5")
        print(
            "Model Precision: {}%".format(
                round(confidence_metrics_entry.precision * 100, 2)
            )
        )
        print(
            "Model Recall: {}%".format(
                round(confidence_metrics_entry.recall * 100, 2)
            )
        )
        print(
            "Model F1 score: {}%".format(
                round(confidence_metrics_entry.f1_score * 100, 2)
            )
        )
        print(
            "Model Precision@1: {}%".format(
                round(confidence_metrics_entry.precision_at1 * 100, 2)
            )
        )
        print(
            "Model Recall@1: {}%".format(
                round(confidence_metrics_entry.recall_at1 * 100, 2)
            )
        )
        print(
            "Model F1 score@1: {}%".format(
                round(confidence_metrics_entry.f1_score_at1 * 100, 2)
            )
        )


# Use the model to make a prediction

# model_id = 'MODEL_ID_HERE'
# file_path = '/local/path/to/file'

from google.cloud import automl_v1beta1 as automl

automl_client = automl.AutoMlClient()

# Create client for prediction service.
prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = automl_client.model_path(
    project_id, compute_region, model_id
)

# Read the file content for prediction.
with open(file_path, "rb") as content_file:
    snippet = content_file.read()

# Set the payload by giving the content and type of the file.
payload = {"text_snippet": {"content": snippet, "mime_type": "text/plain"}}

# params is additional domain-specific parameters.
# currently there is no additional parameters supported.
params = {}
response = prediction_client.predict(model_full_id, payload, params)
print("Prediction results:")
for result in response.payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))
