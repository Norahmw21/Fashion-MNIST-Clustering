import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.datasets import fashion_mnist
from PIL import Image
from scipy.stats import mode

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
X_flat = X.reshape((X.shape[0], -1)) / 255.0  # Flatten + Normalize

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion MNIST Cluster Viewer")

k_value = st.selectbox("🔢 Select number of clusters (k):", [5, 7,10])

pca_50 = PCA(n_components=50, random_state=42)
X_pca_50 = pca_50.fit_transform(X_flat)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_flat)

kmeans = KMeans(n_clusters=k_value, random_state=42)
labels = kmeans.fit_predict(X_pca_50)

cluster_to_label = {}
for cluster_id in range(k_value):
    idx = np.where(labels == cluster_id)[0]
    if len(idx) > 0:
        most_common = mode(y[idx], keepdims=True).mode[0]
        cluster_to_label[cluster_id] = class_names[most_common]

input_mode = st.radio("Select input method:", ["Select from dataset", "Upload image"])

if input_mode == "Select from dataset":
    index = st.slider("Choose image index", 0, len(X) - 1, 0)
    image = X[index]
    input_vector = X_flat[index]
else:
    uploaded_file = st.file_uploader("Upload a 28×28 grayscale image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        image = np.array(image)
        input_vector = image.flatten() / 255.0

if 'input_vector' in locals():
    st.image(image, caption="Input Image", width=150)

    input_pca_50 = pca_50.transform([input_vector])
    cluster_id = kmeans.predict(input_pca_50)[0]
    predicted_class = cluster_to_label.get(cluster_id, "Unknown")
    st.markdown(f"**Predicted Cluster ID:** {cluster_id} → **{predicted_class}**")

    cluster_members = np.where(labels == cluster_id)[0]
    distances = np.linalg.norm(X_pca_50[cluster_members] - input_pca_50, axis=1)
    top_indices = cluster_members[np.argsort(distances)[:5]]

    st.write("🔍 Most similar images in the same cluster:")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in zip(top_indices, axes):
        ax.imshow(X[i], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)

    input_pca_2d = pca_2d.transform([input_vector])
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=1, alpha=0.5)
    ax.scatter(input_pca_2d[0, 0], input_pca_2d[0, 1], color='red', s=80, label='Your Image')
    ax.set_title("PCA Projection (Red = Your Image)")
    ax.legend()
    st.pyplot(fig)
