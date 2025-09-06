import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model("my_model.h5")


# Load x_test and test
x_test = np.load("x_test.npy")
with open("test_labels.pkl", "rb") as f:
    test = pickle.load(f)


# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)




image_index = random.randint(0, len(test))
print("Original Output:", test['label'][image_index])

pred = model.predict(x_test[image_index].reshape(1, 48, 48, 1))
prediction_label = le.inverse_transform([pred.argmax()])[0]
print("Predicted Output:", prediction_label)

plt.imshow(x_test[image_index].reshape(48, 48), cmap='gray')
plt.show()