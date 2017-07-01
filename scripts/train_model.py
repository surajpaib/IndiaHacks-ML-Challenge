import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
try:
    train_X = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/train_X.csv')
    train_Y = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/train_Y.csv')
    test_X = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/test_X.csv')
    label = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/label.csv')
    print "Files read"
except:
    print "Read Error"

train_Y = np.asarray(train_Y['Labels'])
print train_Y.shape
model = GradientBoostingClassifier(n_estimators=800, verbose=True)
param_grid = {
    'max_depth': [5, 8]
}
scorer = make_scorer(roc_auc_score)
cv_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, verbose=True, n_jobs=-1, cv=3)
cv_model.fit(train_X, train_Y)
results = cv_model.predict_proba(test_X)
results = pd.DataFrame(columns=['seg', 'segment'], data=results)
results.drop('seg', axis=1, inplace=True)
results['ID'] = label
results = results[['ID', 'segment']]
print results.head()
results.to_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/submissions/hotstarsub1.csv', index=False)