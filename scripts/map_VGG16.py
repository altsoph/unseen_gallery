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
    # np.bool = np.bool_

    from glob import glob
    import json

    # from tensorflow.keras.applications import VGG16, ResNet50
    # from tensorflow.keras.models import Model

    import torchvision.models as models
    import torchvision.transforms as transforms
    return (
        BytesIO,
        Image,
        glob,
        json,
        models,
        np,
        requests,
        torch,
        transforms,
    )


@app.cell
def __(models):
    # Load the model (example with VGG16)
    # base = VGG16(weights='imagenet', include_top=False)  # include_top=False excludes the final classification layer
    # Load VGG16 pre-trained model
    model = models.vgg16(pretrained=True).features.eval()  # Only the feature layers
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

    # In VGG16, layers like block3_conv3 or block4_conv3 work well for style, while deeper layers capture more semantic details.
    # Their indices are 16 and 23

    # Define a function to extract embeddings
    def extract_embeddings(image_tensor):
        # layer_index=16
        # Pass through the model up to the chosen layer
        with torch.no_grad():
            # features = model[:layer_index](image_tensor)
            features = model[16](image_tensor).flatten().numpy()
            # features2 = model[23](image_tensor).flatten().numpy()
            # .tolist() + \
            #             model[24:25](image_tensor).flatten().numpy().tolist()
        return features # .tolist() + features2.tolist()
        # return features.flatten().numpy()  # Convert to 1D feature vector

    def imagefn2embedding(image_path):
        image_tensor = preprocess_image(image_path)
        return extract_embeddings(image_tensor) # .numpy()[0]

    imagefn2embedding('imgs/6.0000_4b64b121ade4e53e5de04fdc44834352c704b256adc9b45b9793b761f1ad8343_3c00bdf6944a8f13e511b0bedced02801bcdfe19ab7e501baf0ebc380ad96426.jpg')
    return extract_embeddings, imagefn2embedding, preprocess_image


@app.cell
def __(glob, imagefn2embedding, json):
    items = []
    with open('vgg16@16_embds.jsonl', 'w') as ofh:
        for _idx, fn in enumerate(glob('imgs/*.jpg')):
            img_name = fn.replace('\\','/').split('/')[-1]
            _item = {
                'img': img_name,
                'emb': imagefn2embedding(fn).tolist(),
            }
            items.append(_item)
            print(json.dumps(_item), file=ofh, flush=True) # , indent=2
            if not _idx%500:
                print(_idx)
    return fn, img_name, items, ofh


@app.cell
def __(items, np):
    X = []
    for it in items:
        X.append( it['emb'] )
    X = np.array(X)
    X.shape
    return X, it


@app.cell
def __(X, np):
    with open(f'vgg16@16_embs.npy', 'wb') as ff:
    	np.save(ff, np.array(X))
    return (ff,)


@app.cell
def __():
    # items = []
    # X = []
    # for idx, line in enumerate(open('vgg16@16_embds.jsonl')):
    #     item = json.loads(line)
    #     items.append(item)
    #     X.append( np.array(it['emb']) )
    #     if not idx%500:
    #         print(idx)

    # X = np.array(X)
    # X.shape
    return


@app.cell
def __():
    # import importlib

    # # Reimport or reload a module
    # importlib.reload(np)

    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    return TSNE, plt


@app.cell
def __(TSNE, X, np, plt):
    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(X)
    print(X_embedded.shape)
    with open(f'vgg16@16_embs_tsne.npy', 'wb') as f:
    	np.save(f, np.array(X_embedded))

    plt.scatter(*zip(*X_embedded), alpha=.1)
    return X_embedded, f


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
    with open(f'vgg16@16_embs_tsne_u5.npy', 'wb') as of2:
    	np.save(of2, np.array(X_embedded_u5))
    plt.scatter(*zip(*X_embedded_u5), alpha=.05)
    return X_embedded_u5, of2, umap_model


@app.cell
def __():
    from sklearn.cluster import DBSCAN
    import pandas as pd
    return DBSCAN, pd


@app.cell
def __(DBSCAN, X_embedded_u5, pd):

    # Perform DBSCAN clustering
    dbscan4 = DBSCAN(eps=0.08, min_samples=7)  # Adjust eps and min_samples as needed
    cluster_labels2 = dbscan4.fit_predict(X_embedded_u5)

    # Save clusters
    # Combine the embedded data and the cluster labels into a DataFrame for easier handling
    data_with_clusters2 = pd.DataFrame(X_embedded_u5, columns=['x', 'y'])
    data_with_clusters2['cluster'] = cluster_labels2

    # Count the number of items per cluster
    cluster_counts2 = data_with_clusters2['cluster'].value_counts().sort_index()

    # Display the number of items per cluster
    print("Number of items per cluster:")
    print(cluster_counts2)

    data_with_clusters2
    return cluster_counts2, cluster_labels2, data_with_clusters2, dbscan4


@app.cell
def __(cluster_counts2):
    for q in cluster_counts2.items():
        if q[1]>150:
            print(q)
    return (q,)


@app.cell
def __(X_embedded_u5, data_with_clusters2, plt):
    plt.scatter(*zip(*X_embedded_u5), alpha=.01)
    clusters_to_draw = [0, 1, 3, 4,5,6,7,10,14] 
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"]
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
def __():
    # for virtual tour it's okay to use 3,4 or 1  I think
    return


@app.cell
def __(cluster_labels2):
    cluster_labels2
    return


@app.cell
def __(cluster_labels2, json):
    with open('vgg16@16_clusters.json', 'w') as _ofh:
        print(json.dumps(cluster_labels2.tolist()), file=_ofh)

    return


if __name__ == "__main__":
    app.run()
