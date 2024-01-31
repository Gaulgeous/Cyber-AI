import numpy as np
import pandas as pd
import seaborn as sns
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

if __name__=="__main__":
    
    data = pd.read_csv(r"/home/david/Documents/Cyber-AI/data/Training_data_reduced.csv")

    labels = data["Result"]
    data = data.drop("Result", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    pipeline_optimizer = TPOTClassifier(verbosity=1, max_time_mins=10)
    
    pipeline_optimizer.fit(X_train, y_train)

    print(pipeline_optimizer.score(X_test, y_test))

    pipeline_optimizer.export('/home/david/Documents/Cyber-AI/pipelines/tpot_pipeline.py')
