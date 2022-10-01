from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import argparse

def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds

def next_batch(X, y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield(X[i : i+batchSize], y[i : i+batchSize])

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=int, default=0.01, help= "learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of SGD mini-batch")
args = vars(ap.parse_args())

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over desired number of epochs
for epoch in np.arange(args["epochs"]):
    epochLoss = []

    for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):
        preds = sigmoid_activation(batchX.dot(W))

        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        d = error * sigmoid_deriv(preds)
        gradient = batchX.T.dot(d)

        W += -args["alpha"] * gradient

    loss = np.average(epochLoss)
    losses.append(loss)

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()


