import marimo

__generated_with = "0.9.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import torch
    from torchvision import transforms
    from PIL import Image
    import requests
    from io import BytesIO
    import numpy as np
    return BytesIO, Image, np, requests, torch, transforms


@app.cell
def __():
    from glob import glob
    import json
    return glob, json


@app.cell
def __(torch):
    # Load the DINO model from torch.hub
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    model.eval()  # Set model to evaluation mode

    return (model,)


@app.cell
def __(BytesIO, Image, model, requests, torch, transforms):
    # Define a function to preprocess the image
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model's input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load image
        if image_path.startswith('http'):  # If it's a URL, download the image
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")  # Open local image
        
        # Preprocess and return
        return transform(img).unsqueeze(0)  # Add batch dimension

    # Define a function to extract embeddings
    def extract_embeddings(image_tensor):
        with torch.no_grad():
            embeddings = model(image_tensor)
        return embeddings

    def imagefn2embedding(image_path):
        image_tensor = preprocess_image(image_path)
        return extract_embeddings(image_tensor).numpy()[0]

    #imagefn2embedding('imgs/6.0000_4b64b121ade4e53e5de04fdc44834352c704b256adc9b45b9793b761f1ad8343_3c00bdf6944a8f13e511b0bedced02801bcdfe19ab7e501baf0ebc380ad96426.jpg')
    return extract_embeddings, imagefn2embedding, preprocess_image


@app.cell
def __(glob, imagefn2embedding, json):
    items = []
    with open('dyno_embds.jsonl', 'w') as ofh:
        for idx, fn in enumerate(glob('imgs/*.jpg')):
            img_name = fn.replace('\\','/').split('/')[-1]
            item = {
                'img': img_name,
                'emb': imagefn2embedding(fn).tolist(),
            }
            items.append(item)
            print(json.dumps(item), file=ofh, flush=True) # , indent=2
            if not idx%500:
                print(idx)

    return fn, idx, img_name, item, items, ofh


@app.cell
def __(items, np):
    X = []
    for it in items:
        X.append( it['emb'] )
    X = np.array(X)
    X.shape
    return X, it


@app.cell
def __():
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    return TSNE, plt


@app.cell
def __(TSNE, X, np):
    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(X)
    print(X_embedded.shape)
    with open(f'dino_embs_tsne.npy', 'wb') as f:
    	np.save(f, np.array(X_embedded))
    return X_embedded, f


@app.cell
def __(X_embedded, plt):
    plt.scatter(*zip(*X_embedded), alpha=.1)
    return


@app.cell
def __(TSNE, X, np):
    X_embedded_p1 = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=1).fit_transform(X)
    print(X_embedded_p1.shape)
    with open(f'dino_embs_tsne_p1.npy', 'wb') as of:
    	np.save(of, np.array(X_embedded_p1))
    return X_embedded_p1, of


@app.cell
def __(X_embedded_p1, plt):
    plt.scatter(*zip(*X_embedded_p1), alpha=.05)
    return


@app.cell
def __():
    # pip install umap-learn
    import umap
    return (umap,)


@app.cell
def __(X, np, plt, umap):
    umap_model = umap.UMAP(n_components=2, init='random', n_neighbors=5)
    X_embedded_u5 = umap_model.fit_transform(X)
    print(X_embedded_u5.shape)
    with open(f'dino_embs_tsne_u5.npy', 'wb') as of2:
    	np.save(of2, np.array(X_embedded_u5))
    plt.scatter(*zip(*X_embedded_u5), alpha=.05)
    return X_embedded_u5, of2, umap_model


@app.cell
def __(X, np, plt, umap):
    umap_model3 = umap.UMAP(n_components=2, init='random', n_neighbors=3)
    X_embedded_u3 = umap_model3.fit_transform(X)
    print(X_embedded_u3.shape)
    with open(f'dino_embs_tsne_u3.npy', 'wb') as of3:
    	np.save(of3, np.array(X_embedded_u3))
    plt.scatter(*zip(*X_embedded_u3), alpha=.05)
    return X_embedded_u3, of3, umap_model3


@app.cell
def __(X, np, plt, umap):
    umap_model7 = umap.UMAP(n_components=2, init='random', n_neighbors=7)
    X_embedded_u7 = umap_model7.fit_transform(X)
    print(X_embedded_u7.shape)
    with open(f'dino_embs_tsne_u7.npy', 'wb') as of4:
    	np.save(of4, np.array(X_embedded_u7))
    plt.scatter(*zip(*X_embedded_u7), alpha=.05)
    return X_embedded_u7, of4, umap_model7


@app.cell
def __(X, np, plt, umap):
    umap_model4 = umap.UMAP(n_components=2, init='random', n_neighbors=4)
    X_embedded_u4 = umap_model4.fit_transform(X)
    print(X_embedded_u4.shape)
    with open(f'dino_embs_tsne_u4.npy', 'wb') as of5:
    	np.save(of5, np.array(X_embedded_u4))
    plt.scatter(*zip(*X_embedded_u4), alpha=.05)
    return X_embedded_u4, of5, umap_model4


@app.cell
def __(X, np, plt, umap):
    umap_model2 = umap.UMAP(n_components=2, init='random', n_neighbors=2)
    X_embedded_u2 = umap_model2.fit_transform(X)
    print(X_embedded_u2.shape)
    with open(f'dino_embs_tsne_u2.npy', 'wb') as of6:
    	np.save(of6, np.array(X_embedded_u2))
    plt.scatter(*zip(*X_embedded_u2), alpha=.05)
    return X_embedded_u2, of6, umap_model2


@app.cell
def __():
    from sklearn.cluster import DBSCAN
    import pandas as pd
    return DBSCAN, pd


@app.cell
def __(DBSCAN, X_embedded_u4, pd):
    # Assuming X_embedded is your transformed data with shape (14813, 2)
    # Perform DBSCAN clustering
    dbscan4 = DBSCAN(eps=0.5, min_samples=4)  # Adjust eps and min_samples as needed
    cluster_labels4 = dbscan4.fit_predict(X_embedded_u4)

    # Save clusters
    # Combine the embedded data and the cluster labels into a DataFrame for easier handling
    data_with_clusters4 = pd.DataFrame(X_embedded_u4, columns=['x', 'y'])
    data_with_clusters4['cluster'] = cluster_labels4
    return cluster_labels4, data_with_clusters4, dbscan4


@app.cell
def __(DBSCAN, X_embedded_u5, X_embedded_u7, pd):
    # Assuming X_embedded is your transformed data with shape (14813, 2)
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=7)  # Adjust eps and min_samples as needed
    cluster_labels = dbscan.fit_predict(X_embedded_u7)

    # Save clusters
    # Combine the embedded data and the cluster labels into a DataFrame for easier handling
    data_with_clusters = pd.DataFrame(X_embedded_u5, columns=['x', 'y'])
    data_with_clusters['cluster'] = cluster_labels
    return cluster_labels, data_with_clusters, dbscan


@app.cell
def __(data_with_clusters4):
    # Count the number of items per cluster
    cluster_counts4 = data_with_clusters4['cluster'].value_counts().sort_index()

    # Display the number of items per cluster
    print("Number of items per cluster:")
    print(cluster_counts4)

    data_with_clusters4
    return (cluster_counts4,)


@app.cell
def __(X_embedded_u2, dbscan4, pd):
    # Assuming X_embedded is your transformed data with shape (14813, 2)
    # Perform DBSCAN clustering
    cluster_labels2 = dbscan4.fit_predict(X_embedded_u2)

    # Save clusters
    # Combine the embedded data and the cluster labels into a DataFrame for easier handling
    data_with_clusters2 = pd.DataFrame(X_embedded_u2, columns=['x', 'y'])
    data_with_clusters2['cluster'] = cluster_labels2

    # Count the number of items per cluster
    cluster_counts2 = data_with_clusters2['cluster'].value_counts().sort_index()

    # Display the number of items per cluster
    print("Number of items per cluster:")
    print(cluster_counts2)

    data_with_clusters2
    return cluster_counts2, cluster_labels2, data_with_clusters2


@app.cell
def __(cluster_counts2):
    for q in cluster_counts2.items():
        if q[1]>150:
            print(q)
        
    return (q,)


@app.cell
def __(X_embedded_u2, data_with_clusters2, plt):
    plt.scatter(*zip(*X_embedded_u2), alpha=.01)
    clusters_to_draw = [1, 4,7,9,16] 
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"]
    for i, c in enumerate(clusters_to_draw):
        subset_data = data_with_clusters2[data_with_clusters2['cluster'].isin([c,])]
        plt.scatter(subset_data['x'], subset_data['y'], c=colors[i], label=c) # subset_data['cluster'], cmap='viridis', s=5 
    plt.legend()
    plt.show()
    # subset_data
    return c, clusters_to_draw, colors, i, subset_data


@app.cell
def __(cluster_labels2, clusters_to_draw, data_with_clusters2, items, np):
    # Display random examples with indices from each cluster
    cluster_examples_with_indices = {}
    for cluster in np.unique(cluster_labels2):
        if cluster == -1:
            # Skip noise points labeled as -1 in DBSCAN
            continue
        # Filter examples in the cluster and sample up to 10, keeping the index
        examples = data_with_clusters2[data_with_clusters2['cluster'] == cluster].sample(n=min(20, sum(data_with_clusters2['cluster'] == cluster)), random_state=42)
        cluster_examples_with_indices[cluster] = examples.index.tolist()  # Store indices

    # Display cluster examples with indices
    with open('clusters_showcase.html', 'w') as ohtmlfh:
        for cluster, indices in cluster_examples_with_indices.items():
            if cluster not in clusters_to_draw: continue
            print(f"<h1>Cluster {cluster} - Example Indices (up to 10):</h1>", file=ohtmlfh)
            print(f"<h2>  {indices}<h2>", file=ohtmlfh)
            for _idx in indices:
                print(f"<img src='./imgs/{items[_idx]['img']}' width=300 height=300/>", file=ohtmlfh)
    return (
        cluster,
        cluster_examples_with_indices,
        examples,
        indices,
        ohtmlfh,
    )


@app.cell
def __(mo):
    mo.md(f'<img src="./imgs/6.0000_4b64b121ade4e53e5de04fdc44834352c704b256adc9b45b9793b761f1ad8343_3c00bdf6944a8f13e511b0bedced02801bcdfe19ab7e501baf0ebc380ad96426.jpg"/>')
    #alt="Image" width="300" height="300">')
    return



if __name__ == "__main__":
    app.run()
