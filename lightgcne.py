import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.datasets import AmazonBook
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree
from torch_geometric.data import Data
from collections import Counter
import math

import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

device = torch.device('cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazonbook-5k')
dataset = AmazonBook(path)
data = dataset[0]
num_users, num_books = data['user'].num_nodes, data['book'].num_nodes
data = data.to_homogeneous().to(device)
print(num_users, num_books)
# Use all message passing edges as training labels:
batch_size = 8192
mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]


train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size,
)

model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=64,
    num_layers=1,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_books,
                          (index.numel(), ), device=device)
        ], dim=0)
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples


def compute_evgin(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.numpy()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))


    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    n = []
    for node, value in  eigenvector_centrality.items():
        n.append(value)
    arr = np.array(n)
    arr[arr > 0] = 1 + arr[arr > 0]
    arr[arr < 0] = arr[arr < 0] - 1
    nt = torch.from_numpy(arr)
    torch.save(nt,"amazon5k.pt")

def train_e(edge_weight):
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_books,
                          (index.numel(), ), device=device)
        ], dim=0)
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index,edge_weight).chunk(2)

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples

@torch.no_grad()
def test(k: int):
    emb = model.get_embedding(data.edge_index)
    user_emb, book_emb = emb[:num_users], emb[num_users:]

    precision = recall = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ book_emb.t()

        # Exclude training edges:
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # Computing precision and recall:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples


def compute_tf_idf(numbers):
    total_numbers = len(numbers)
    tf = {num: count / total_numbers for num, count in Counter(numbers).items()}

    unique_numbers = set(numbers)
    idf = {num: math.log(total_numbers / (count + 1)) for num, count in Counter(numbers).items()}

    tf_idf = {num: tf[num] * idf[num] for num in unique_numbers}
    return tf_idf


def replace_with_weights(tensor, weight_dict):
    return np.vectorize(weight_dict.get, otypes=[float])(tensor, 0)


if __name__ == "__main__":

    compute_evgin(data)
    weight = torch.load("amazon5k.pt")

    row, col = data.edge_index
    edge_weight1 = weight[row]

    r_recall = 0
    p_recall = 0
    #LightGCN-E
    for j in range(0,5):
        s = 0
        for epoch in range(0, 200):
            loss = train_e(edge_weight1)
            precision, recall = test(k=20)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
                    f'{precision:.4f}, Recall@20: {recall:.4f}')

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)

    # LightGCN
    # for j in range(0, 5):
    #
    #     for epoch in range(0, 200):
    #
    #         # loss = train()
    #         precision, recall = test(k=20)
    #         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
    #               f'{precision:.4f}, Recall@20: {recall:.4f}')
    #
    #     model.reset_parameters()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #     optimizer.zero_grad(set_to_none=True)
