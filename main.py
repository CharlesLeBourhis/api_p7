from fastapi import FastAPI
from deta import Drive, Deta
import numpy as np
import pandas as pd
import pickle
from io import BytesIO
import os


deta = Deta(os.environ["DETA_KEY"])
drive = deta.Drive("api")

model_drive = drive.get("model_lgbm_03.pkl")
model_bytes = model_drive.read()
model_drive.close()

model = pickle.loads(model_bytes)


cluster_drive = drive.get('cluster.pkl')
cluster_bytes = cluster_drive.read()
cluster_drive.close()

clustering = pickle.loads(cluster_bytes)


df = pd.read_csv("https://raw.githubusercontent.com/CharlesLeBourhis/Projet7/main/df_sample.csv")

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
    client = df.query("SK_ID_CURR == @sk_id").drop("SK_ID_CURR", axis=1)
    score = model.predict_proba(client.values.reshape(1, -1))[0,0]
    return {"score": score,  "client_df": client.to_json()}