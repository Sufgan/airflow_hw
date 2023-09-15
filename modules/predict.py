import dill
import os
import pandas as pd
import json
import datetime as dt

path = os.environ.get('PROJECT_PATH', '.')

def predict():
    model = load_model()

    pred_df = pd.DataFrame(columns=['car_id', 'prediction'])

    for file in [f.name for f in os.scandir(f'{path}/data/test/')]:
        text = open(f'{path}/data/test/{file}').read()
        df = pd.DataFrame(json.loads(text), index=[0])

        prediction = pd.DataFrame({'car_id' : df.id, 'prediction' : [model.predict(df)[0]]})
        pred_df = pd.concat([pred_df, prediction], ignore_index=True)
    
    save_df(pred_df)

def load_model():
    with open(f'{path}/data/models/' + [f.name for f in os.scandir(f'{path}/data/models/')][0], 'rb') as file:
        return dill.load(file)

def save_df(df):
    df.to_csv(f'{path}/data/predictions/pred_{dt.datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)
    

if __name__ == '__main__':
    predict()
