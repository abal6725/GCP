import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

if __name__ == '__main__':
  content = sys.argv[1]
  project_id = sys.argv[2]
  model_id = sys.argv[3]

  print (get_prediction(content, project_id,  model_id))

x = get_prediction(data.loc[10,'CleanedTweets'],'global-brook-217321','TCN7983166574825777366')
