import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

try:
    train_X = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/train_X.csv')
    train_Y = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/train_Y.csv')
    test_X = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/test_X.csv')
    label = pd.read_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/label.csv')
    print "Files read"
except:
    print "Read Error"

model = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_depth=5, verbose=True)
model.fit(train_X, train_Y)
results = model.predict_proba(test_X)
results = pd.DataFrame(columns=['seg', 'segment'], data=results)
results.drop('seg', axis=1, inplace=True)
results['ID'] = label
results = results[['ID', 'segment']]
print results.head()
results.to_csv('/home/suraj/Repositories/IndiaHacks ML Hackathon/submissions/hotstarsub1.csv', index=False)