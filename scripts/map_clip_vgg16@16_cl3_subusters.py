import marimo

__generated_with = "0.9.15"
app = marimo.App(width="medium")


@app.cell
def __():
    # import marimo as mo
    return


@app.cell
def __():
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from torchvision import transforms
    import requests
    from io import BytesIO
    import numpy as np

    from glob import glob
    import json

    from sklearn.manifold import TSNE
    import umap
    from matplotlib import pyplot as plt
    from sklearn.cluster import DBSCAN
    import pandas as pd

    from collections import defaultdict, Counter
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors
    return (
        BytesIO,
        CLIPModel,
        CLIPProcessor,
        Counter,
        DBSCAN,
        Image,
        NearestNeighbors,
        TSNE,
        defaultdict,
        glob,
        json,
        np,
        nx,
        pd,
        plt,
        requests,
        torch,
        transforms,
        umap,
    )


@app.cell
def __(CLIPModel, CLIPProcessor):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor


@app.cell
def __(BytesIO, Image, model, processor, requests, torch):
    # Function to process and get embeddings from an image
    def imagefn2embedding(image_path):
        if image_path.startswith("http"):  # If it's a URL, download the image
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:  # Otherwise, load the local image
            image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
        return image_embeddings[0].numpy()

    imagefn2embedding('imgs/6.0000_4b64b121ade4e53e5de04fdc44834352c704b256adc9b45b9793b761f1ad8343_3c00bdf6944a8f13e511b0bedced02801bcdfe19ab7e501baf0ebc380ad96426.jpg')
    return (imagefn2embedding,)


@app.cell
def __():
    return


@app.cell
def __(json):
    clusters = json.loads(open('vgg16@16_clusters.json').read())
    print(len(clusters), clusters[:5])
    return (clusters,)


@app.cell
def __(clusters, imagefn2embedding, json):
    items = []
    clusters_to_use = [3,4, 1]

    for _idx, _line in enumerate(open('dyno_embds.jsonl')):
        _item = json.loads(_line)
        _cl = clusters[_idx]
        if _cl in clusters_to_use:
            _item['vis_cluster_id'] = _cl
            del _item['emb']
            _item['clip_emb'] = imagefn2embedding(f'imgs/{_item["img"]}').tolist()
            items.append( _item )

    with open('vgg16@16_cl1_3_4_clip_embeds.jsonl', 'w') as _ofh:
        print(json.dumps(items), file=_ofh)
    return clusters_to_use, items


@app.cell
def __():
    return


@app.cell
def __(TSNE, items, np, plt):
    _target_cluster = 3

    _X = np.array([_it['clip_emb'] for _it in items if _it['vis_cluster_id']==_target_cluster])

    _X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(_X)

    plt.scatter(*zip(*_X_embedded), alpha=.1)
    return


@app.cell
def __(items, np, plt, umap):
    _target_cluster = 3

    _X = np.array([_it['clip_emb'] for _it in items if _it['vis_cluster_id']==_target_cluster])

    _X_embedded = umap.UMAP(n_components=2, init='random', n_neighbors=6).fit_transform(_X)

    plt.scatter(*zip(*_X_embedded), alpha=.1)
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(DBSCAN, items, np, pd, plt, umap):
    _target_cluster = 3

    _X = np.array([_it['clip_emb'] for _it in items if _it['vis_cluster_id']==_target_cluster])

    _X_embedded = umap.UMAP(n_components=2, init='random', n_neighbors=6).fit_transform(_X)

    # plt.scatter(*zip(*_X_embedded), alpha=.1)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.26, min_samples=10)  # Adjust eps and min_samples as needed
    _cluster_labels = dbscan.fit_predict(_X_embedded)

    # Save clusters
    # Combine the embedded data and the cluster labels into a DataFrame for easier handling
    _item2cluster = pd.DataFrame(_X_embedded, columns=['x', 'y'])
    _item2cluster['cluster'] = _cluster_labels
    _cluster_counts = _item2cluster['cluster'].value_counts().sort_index()

    # Display the number of items per cluster
    print("Number of items per cluster:")
    print(_cluster_counts)

    # for q in _cluster_counts.items():
    #     if q[1]>150:
    #         print(q)
    # print()
    plt.scatter(*zip(*_X_embedded), alpha=.1)        
    # plt.scatter(*zip(*X_embedded_u5), alpha=.01)
    # clusters_to_draw = [0, 1, 3, 4,5,6,7,10,14] 
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"]
    colors = colors+colors +colors+colors+colors+colors+colors+colors+colors+colors+colors
    for i, c in enumerate(list(_cluster_counts.keys())[1:]):
        subset_data = _item2cluster[_item2cluster['cluster'].isin([c,])]
        plt.scatter(subset_data['x'], subset_data['y'], c=colors[i], label=c) # subset_data['cluster'], cmap='viridis', s=5 
    # plt.legend()
    plt.show()
    # # subset_data

    item2cluster = _item2cluster
    return c, colors, dbscan, i, item2cluster, subset_data


@app.cell
def __():
    return


@app.cell
def __(item2cluster):
    item2cluster
    return


@app.cell
def __():
    return


@app.cell
def __(defaultdict, item2cluster, items):
    _target_cluster = 3

    _subcluster2imgs = defaultdict(list)
    img2xy = dict()

    for _idx, (_it, _loc_cluster, _x, _y) in enumerate(zip(
        [_it for _it in items if _it['vis_cluster_id']==_target_cluster],
        item2cluster['cluster'].tolist(),
        item2cluster['x'].tolist(),
        item2cluster['y'].tolist(),
    )):
        # print(_idx, _it['img'], _loc_cluster)
        # print(_idx, _x, _y, _loc_cluster)
        _subcluster2imgs[_loc_cluster].append( _it['img'] )
        img2xy[ _it['img'] ] = (_x, _y)

    gallery = []

    with open('vgg16@16_cl3_subusters_showcase.html', 'w') as _ohtmlfh:
        for cluster, indices in _subcluster2imgs.items():
            if cluster == -1: continue
            print(f"<h1>Cluster {cluster} - Total Indices {len(indices)}:</h1>", file=_ohtmlfh)
            # print(f"<h2>  {indices}<h2>", file=_ohtmlfh)
            print(f"<h2>From: {min(indices)}<h2>", file=_ohtmlfh)
            print(f"<h2>To: {max(indices)}<h2>", file=_ohtmlfh)
            gallery.append( max(indices) )
            for _idx in sorted(indices):
                print(f"<img src='./imgs/{_idx}' width=300 height=300/>", file=_ohtmlfh)
    return cluster, gallery, img2xy, indices


@app.cell
def __(gallery):
    gallery
    return


@app.cell
def __(NearestNeighbors, gallery, img2xy, np, nx, plt):
    # _names = [_i[:4] for _i in gallery[:]]
    _names = gallery[:]
    _coordinates = np.array([(img2xy[_img][0], img2xy[_img][1]) for _img in gallery])

    # Step 1: Find minimal k that creates a connected graph
    _k = 2
    while True:
        # Perform KNN for the current k
        _nbrs = NearestNeighbors(n_neighbors=_k).fit(_coordinates)
        _distances, _indices = _nbrs.kneighbors(_coordinates)
        
        # Create a graph from the KNN result
        _G = nx.Graph()
        for _i, _neighbors in enumerate(_indices):
            for _j in _neighbors:
                if _i != _j:  # Avoid self-loop
                    _G.add_edge(_names[_i], _names[_j], weight=_distances[_i][np.where(_neighbors == _j)[0][0]])
        
        # Check if the graph is connected
        if nx.is_connected(_G):
            break  # Stop if we have a connected graph
        _k += 1

    print(f"Connected graph achieved with k = {_k}")

    # Step 2: Try to find a Hamiltonian path (or an approximation)
    # NetworkX has an `approximation` module with a tsp method we can use with default approximation
    try:
        _hamiltonian_path = nx.approximation.traveling_salesman_problem(_G, cycle=False)
    except nx.NetworkXNoPath:
        print("No Hamiltonian path found, proceeding with nearest-neighbor approximation.")
        _hamiltonian_path = None  # No Hamiltonian path found

    # Draw the KNN graph
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    _pos = {name: coord for name, coord in zip(_names, _coordinates)}
    nx.draw(_G, _pos, with_labels=False, node_size=700, node_color='skyblue', font_size=12, font_weight='bold')
    plt.title(f"KNN Graph with k={_k}")

    # Draw the Hamiltonian path (or approximate path)
    if _hamiltonian_path:
        _path_edges = [(_hamiltonian_path[i], _hamiltonian_path[i + 1]) for i in range(len(_hamiltonian_path) - 1)]
        plt.subplot(1, 2, 2)
        nx.draw(_G, _pos, with_labels=False, node_size=700, node_color='lightgreen', font_size=12, font_weight='bold')
        nx.draw_networkx_edges(_G, _pos, edgelist=_path_edges, edge_color='orange', width=2)
        plt.title("Approximate Hamiltonian Path")

    plt.tight_layout()
    plt.show()

    with open('vgg16@16_cl3_subusters_walk.txt', 'w') as _ohtmlfh:
        for _step in _hamiltonian_path:
            print(_step)
            print(_step, file=_ohtmlfh)
        
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
