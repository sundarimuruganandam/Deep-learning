from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np

faces = fetch_olivetti_faces()
X, y = faces.images.reshape(-1, 64, 64, 1).astype('float32'), to_categorical(faces.target, 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)), Flatten(),
    Dense(256, activation='relu'),
    Dense(40, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

output:
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/3ce7d7ef-94b0-42b0-a31d-bdbc9970264b" />
