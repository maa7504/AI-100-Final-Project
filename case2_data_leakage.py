import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
n = 800

study_hours = np.random.uniform(0, 20, n)
attendance = np.random.uniform(50, 100, n)
practice = np.random.uniform(0, 100, n)
sleep = np.random.uniform(4, 10, n)
gpa = np.random.uniform(2.0, 4.0, n)

score = (0.3*study_hours +
         0.01*attendance +
         0.02*practice +
         0.2*sleep +
         2*gpa +
         np.random.normal(0, 2, n))

y = (score > 15).astype(int)
X = np.column_stack((study_hours, attendance, practice, sleep, gpa))

X_train, X_test, y_train, y_test = X, X, y, y

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=5))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=16)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
