import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, auc, precision_recall_curve
from torch_sparse import SparseTensor
from model import GCN_mgaev3 as GCN
from model import SAGE_mgaev2 as SAGE
from model import GIN_mgaev2 as GIN
from model import LPDecoder
from utils import do_edge_split_direct, edgemask_um
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
import time
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt


def draw_auc(y, pred):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('MNDR', fontsize=18)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def draw_aupr(y, pred):
    average_precision = average_precision_score(y, pred)
    precision, recall, _ = precision_recall_curve(y, pred)
    plt.plot(recall, precision, label='AUPR = %0.4f' % average_precision)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('MNDR', fontsize=18)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def tensor_to_numpy(tensor):
    # Converts PyTorch tensors to numpy arrays for metric computation
    return tensor.detach().cpu().numpy()


def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true, draw, threshold=0.85):
    # Convert tensors to numpy arrays for sklearn metrics
    train_pred_np = tensor_to_numpy(train_pred)
    val_pred_np = tensor_to_numpy(val_pred)
    test_pred_np = tensor_to_numpy(test_pred)
    train_true_np = tensor_to_numpy(train_true)
    val_true_np = tensor_to_numpy(val_true)
    test_true_np = tensor_to_numpy(test_true)

    # AUC
    train_auc = roc_auc_score(train_true_np, train_pred_np)
    valid_auc = roc_auc_score(val_true_np, val_pred_np)
    test_auc = roc_auc_score(test_true_np, test_pred_np)
    if draw:
        # draw_auc(test_true_np, test_pred_np)
        draw_aupr(test_true_np, test_pred_np)

    # Average Precision
    train_ap = average_precision_score(train_true_np, train_pred_np)
    valid_ap = average_precision_score(val_true_np, val_pred_np)
    test_ap = average_precision_score(test_true_np, test_pred_np)

    # Convert probabilities to binary predictions based on the threshold
    train_pred_binary = (train_pred > threshold).int()
    val_pred_binary = (val_pred > threshold).int()
    test_pred_binary = (test_pred > threshold).int()

    # Convert binary predictions to numpy for sklearn metrics
    train_pred_binary_np = tensor_to_numpy(train_pred_binary)
    val_pred_binary_np = tensor_to_numpy(val_pred_binary)
    test_pred_binary_np = tensor_to_numpy(test_pred_binary)

    # Accuracy
    train_accuracy = accuracy_score(train_true_np, train_pred_binary_np)
    valid_accuracy = accuracy_score(val_true_np, val_pred_binary_np)
    test_accuracy = accuracy_score(test_true_np, test_pred_binary_np)

    # Precision, Recall, and F1 Score (handling cases where there are no positive labels)
    train_precision = precision_score(train_true_np, train_pred_binary_np, zero_division=0)
    valid_precision = precision_score(val_true_np, val_pred_binary_np, zero_division=0)
    test_precision = precision_score(test_true_np, test_pred_binary_np, zero_division=0)

    train_recall = recall_score(train_true_np, train_pred_binary_np, zero_division=0)
    valid_recall = recall_score(val_true_np, val_pred_binary_np, zero_division=0)
    test_recall = recall_score(test_true_np, test_pred_binary_np, zero_division=0)

    train_f1 = f1_score(train_true_np, train_pred_binary_np, zero_division=0)
    valid_f1 = f1_score(val_true_np, val_pred_binary_np, zero_division=0)
    test_f1 = f1_score(test_true_np, test_pred_binary_np, zero_division=0)

    # Compile results
    results = {
        'AUC': (train_auc, valid_auc, test_auc),
        'AP': (train_ap, valid_ap, test_ap),
        'Accuracy': (train_accuracy, valid_accuracy, test_accuracy),
        'Precision': (train_precision, valid_precision, test_precision),
        'Recall': (train_recall, valid_recall, test_recall),
        'F1': (train_f1, valid_f1, test_f1)
    }

    return results


def train(model, predictor, data, split_edge, optimizer, args):
    model.train()
    predictor.train()

    adj, edge_index, edge_index_mask = edgemask_um(args.mask_ratio, split_edge, data.x.device, data.num_nodes)
    pre_edge_index = adj.to(data.x.device)
    pos_train_edge = edge_index_mask

    optimizer.zero_grad()
    h = model(data.x, pre_edge_index)
    edge = pos_train_edge


   # Get output, mu, and logvar from the predictor
    pos_out, mu, logvar = predictor(h, edge)
    pos_loss = -torch.log(pos_out + 1e-15).mean()

    new_edge_index, _ = add_self_loops(edge_index.cpu())
    edge = negative_sampling(
        new_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=pos_train_edge.shape[1])

    edge = edge.to(data.x.device)


    # Get only the output for negative edges
    neg_out, _, _ = predictor(h, edge)
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()


     # Calculate KL Divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())



     # Add KL Divergence to the loss
    beta = 0.1  # Example beta value
    loss = pos_loss + neg_loss + beta * kl_div

    loss.backward()


    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

    optimizer.step()

    return loss.item()



@torch.no_grad()
def test(model, predictor, data, adj, split_edge, batch_size, draw=False):
    model.eval()
    h = model(data.x, adj)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h, edge)[0].squeeze().cpu()]  # Get the first output
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h, edge)[0].squeeze().cpu()]  # Get the first output
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h, edge)[0].squeeze().cpu()]  # Get the first output
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h, edge)[0].squeeze().cpu()]  # Get the first output
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge)[0].squeeze().cpu()]  # Get the first output
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge)[0].squeeze().cpu()]  # Get the first output
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    train_true = torch.cat([torch.ones_like(pos_train_pred), torch.zeros_like(neg_train_pred)], dim=0)

    val_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    val_true = torch.cat([torch.ones_like(pos_valid_pred), torch.zeros_like(neg_valid_pred)], dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)

    results = evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true, draw)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default='data2')
    parser.add_argument('--de_v', type=str, default='v1', help='v1 | v2')  # whether to use mask features
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--decode_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--decode_channels', type=int, default=512)  # data1:256
    parser.add_argument('--dropout', type=float, default=0.5)  # data1:0.2
    parser.add_argument('--batch_size', type=int, default=64)  # data1:128
    parser.add_argument('--lr', type=float, default=0.01)  # data1:0.005
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mask_ratio', type=float, default=0.2)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50,
                        help='Use attribute or not')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # data1:42
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda:0'
    device = torch.device(device)
    # lncRNADisease MNDR
    disease_feature = pd.read_excel(
        "dataset/{}/MNDR-disease semantic similarity matrix.xls".format(args.dataset),
        header=0, index_col=0).values.astype(np.float32)
    lncRNA_feature = pd.read_excel(
        "dataset/{}/MNDR-lncRNA functional similarity matrix.xls".format(args.dataset),
        header=0, index_col=0).values.astype(np.float32)
    adj = pd.read_excel(
        "dataset/{}/MNDR-lncRNA-disease associations matrix.xls".format(args.dataset),
        header=0, index_col=0).values

    lncRNA_feature_padded = np.pad(lncRNA_feature, ((0, 0), (0, 256 - len(lncRNA_feature))), 'constant',
                                   constant_values=(0,))
    disease_feature_padded = np.pad(disease_feature, ((0, 0), (0, 256 - len(disease_feature))), 'constant',
                                    constant_values=(0,))
    features = np.concatenate((lncRNA_feature_padded, disease_feature_padded), axis=0)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    edge_index = np.nonzero(adj)
    edge_index_tensor = torch.LongTensor(edge_index)
    edge_index_tensor[1] = edge_index_tensor[1] + len(lncRNA_feature)

    data = Data(x=features_tensor, edge_index=edge_index_tensor)

    split_edge = do_edge_split_direct(data)

    data.edge_index = to_undirected(split_edge['train']['edge'].t())
    if args.use_sage == 'GCN':
        edge_index, _ = add_self_loops(data.edge_index)
        adj = SparseTensor.from_edge_index(edge_index).t()
    else:
        edge_index = data.edge_index
        adj = SparseTensor.from_edge_index(edge_index).t()

    data = data.to(device)
    adj = adj.to(device)

    print('Start training with mask ratio={} # optimization edges={} / {}'.format(args.mask_ratio,
                                                                                  int(args.mask_ratio *
                                                                                      split_edge['train']['edge'].shape[
                                                                                          0]),
                                                                                  split_edge['train']['edge'].shape[0]))

    metric = 'AUC'
    if args.use_sage == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.use_sage == 'GIN':
        model = GIN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.use_sage == 'GCN':
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    predictor = VAEDecoder(args.hidden_channels, args.decode_channels, 1, args.num_layers,
                          args.decode_layers, args.dropout, latent_dim=64, de_v=args.de_v).to(device) # Add latent_dim here

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer, args)
            t2 = time.time()

            results = test(model, predictor, data, adj, split_edge, args.batch_size)
            # print(results)
            valid_hits = results[metric][1]
            if valid_hits > best_valid:
                best_valid = valid_hits
                best_epoch = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1

            # for key, result in results.items():
            #     train_hits, valid_hits, test_hits = result
            #     print(key)
            #     print(f'Run: {run + 1:02d} / {args.runs:02d}, '
            #           f'Epoch: {epoch:02d} / {args.epochs + 1:02d}, '
            #           f'Best_epoch: {best_epoch:02d}, '
            #           f'Best_valid: {100 * best_valid:.2f}%, '
            #           f'Loss: {loss:.4f}, '
            #           f'Train: {100 * train_hits:.2f}%, '
            #           f'Valid: {100 * valid_hits:.2f}%, '
            #           f'Test: {100 * test_hits:.2f}%',
            #           f'Time: {t2 - t1:.2f}%')
            # print('***************')
            if cnt_wait == args.patience:
            #     print('Early stopping!')
                break
        # print('##### Testing on {}/{}'.format(run, args.runs))

        results = test(model, predictor, data, adj, split_edge, args.batch_size, draw=True)
        print(results)


if __name__ == "__main__":
    main()
    plt.show()
