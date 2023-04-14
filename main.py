from fastapi import FastAPI
from deta import Drive, Deta
import numpy as np
import pandas as pd
import pickle
from io import BytesIO

deta = Deta("a0ffpxdsgpe_L24sS9B5wFaP5YZYPBB1VW3kWcin8ax5")
drive = deta.Drive("api")

model_drive = drive.get("pipeline_lr_02.pkl")
model_bytes = model_drive.read()
model_drive.close()

model = pickle.loads(model_bytes)


cluster_drive = drive.get('cluster.pkl')
cluster_bytes = cluster_drive.read()
cluster_drive.close()

clustering = pickle.loads(cluster_bytes)


feat_drive = drive.get('selected_features_01.pkl')
feat_bytes = feat_drive.read()
feat_drive.close()

selected_features = pickle.loads(feat_bytes)


data_drive = drive.get('data_combined_05.parquet')
data_bytes = data_drive.read()
data_drive.close()

df = pd.read_parquet(BytesIO(data_bytes))

app = FastAPI()

@app.get("/")
def home():
    return {"test":"OK"}

@app.post("/predict")
def predict(client):
    client = pd.read_json(client, typ="series")
    score = model.predict_proba(client.values.reshape(1, -1))[0, 0]
    return {"score": score}


@app.post("/cluster")
def cluster(client):
    client = pd.read_json(client, typ="series")
    cluster = clustering.predict(client.replace({np.inf:0, -np.inf:0}).fillna(-1).values.reshape(1, -1))
    return {"cluster": cluster.tolist()}

@app.post("/client")
def client(Id):
    sk_id = int(Id)
    client = df.query("SK_ID_CURR == @sk_id")
    score = model.predict_proba(client[selected_features].values.reshape(1, -1))[0,0]
    return {"score": score,  "client_df": client.to_json()}