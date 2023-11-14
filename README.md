import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

file_path = r'C:/Farizka/Lenovo/Documents/another_data.csv'
data = pd.read_csv(file_path)

selected_columns = ['target', 'feature1', 'feature2', 'category']
data = data[selected_columns]

data.fillna(0, inplace=True)
data = pd.get_dummies(data)

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = RandomForestRegressor()
model2 = Ridge()
model3 = KNeighborsRegressor()

ensemble = VotingRegressor(estimators=[('rf', model1), ('ridge', model2), ('knn', model3)])
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Voting Regressor: {mse}')
