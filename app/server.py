from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib
import json
import time


from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "../data/news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "./data/logs.out"
    }
}

class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    #estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config
        
        """
        [TO BE IMPLEMENTED]
        1. Load the sentence transformer model and initialize the `featurizer` of type `TransformerFeaturizer` (Hint: revisit Week 1 Step 4)
        2. Load the serialized model as defined in GLOBAL_CONFIG['model'] into memory and initialize `model`
    """
        sentence_transformer_model = SentenceTransformer(config["model"]["featurizer"]["sentence_transformer_model"])
        
        featurizer = TransformerFeaturizer(dim=config["model"]["featurizer"]["sentence_transformer_embedding_dim"], sentence_transformer_model=sentence_transformer_model)

        model = joblib.load(config["model"]["classifier"]["serialized_model_path"])
        self.class_names = model.classes_

        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', model)
         ])
              


    def predict_proba(self, model_input: dict) -> dict:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization, 
        run model inference on a given model input, and return the 
        model prediction probability scores across all labels

        Output format: 
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2 
            ...
        }
        """

        classifier = joblib.load(self.config["model"]["classifier"]["serialized_model_path"])
        class_names = classifier.classes_
        pred_proba = self.pipeline.predict_proba([model_input])
        pred_label_scores = {
            class_names[i]: pred_proba[0][i]
            for i in range(len(class_names))
        }

        return pred_label_scores


    def predict_label(self, model_input: dict) -> str:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """

        prob = self.pipeline.predict_proba([model_input])
        pred_label_scores = {
            self.class_names[i]: prob[0][i]
            for i in range(len(self.class_names))
        }
        pred_label = self.class_names[prob[0].argmax()]
        return pred_label


app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
        [TO BE IMPLEMENTED]
        2. Initialize the `NewsCategoryClassifier` instance to make predictions online. You should pass any relevant config parameters from `GLOBAL_CONFIG` that are needed by NewsCategoryClassifier 
        3. Open an output file to write logs, at the destimation specififed by GLOBAL_CONFIG['service']['log_destination']
        
        Access to the model instance and log file will be needed in /predict endpoint, make sure you
        store them as global variables
    """

    global classifier

    classifier = NewsCategoryClassifier(GLOBAL_CONFIG)

    logger.remove()
    logger.add(GLOBAL_CONFIG["service"]["log_destination"], format="{message}")

    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
        [TO BE IMPLEMENTED]
        1. Make sure to flush the log file and close any file pointers to avoid corruption
        2. Any other cleanups
    """
    logger.info("Shutting down application")
    logger.remove()
   
    


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
        [TO BE IMPLEMENTED]
        1. run model inference and get model predictions for model inputs specified in `request`
        2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`, and writes to the path defined in GLOBAL_CONFIG['service']['log_destination'])
        {
            'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
            'request': dictionary representation of the input request,
            'prediction': dictionary representation of the response,
            'latency': time it took to serve the request, in millisec
        }
        3. Construct an instance of `PredictResponse` and return
    """

    start_time = time.time()

    p_label = classifier.predict_label(request.description)
    scores = classifier.predict_proba(request.description)

    completion_time = time.time()    
    latency = completion_time - start_time



    response = {"scores": scores, "label": p_label}


    log_ = {
        'timestamp': time.strftime("%Y:%m:%d, %H:%M:%S"),
        'request': request,
        'prediction': response, 
        'latency': latency,
    }
    logger.info(log_)
    
   
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
