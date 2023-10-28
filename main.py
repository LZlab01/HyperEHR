import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from sklearn.utils import class_weight
from utils import setup_seed, print_metrics_binary, get_median, MakeEdge, device
from model import Model
from data_extraction import data_process_eICU


def train(train_gen,
          train_steps,
          valid_gen,
          valid_steps,
          in_dim,
          hidden_dim,
          phi,
          gcn_dim,
          MLP_dims,
          N,
          drop_prob,
          lr,
          seed,
          epochs,
          file_name,
          device):

    model = Model(in_dim, hidden_dim, phi, gcn_dim, MLP_dims, N, drop_prob).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_model, milestones=[40, 60, 80, 90], gamma=0.5)

    setup_seed(seed)
    train_loss = []
    best_epoch = 0
    max_auroc = 0

    for each_epoch in range(epochs):
        batch_loss = []
        model.train()

        for step in range(train_steps):
            ret = next(train_gen)
            x = [np.hstack((ret[0][0][i], ret[0][1][i])) for i in range(len(ret[0][0]))]
            y = ret[1]
            static = ret[2]
            sorted_length = ret[3]
            batch_x = torch.FloatTensor(x).to(device)
            batch_y = torch.LongTensor(y).squeeze().to(device)

            x_mean = torch.stack(get_median(batch_x)).to(device)
            x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
            batch_x = torch.where(batch_x == -1, x_mean, batch_x)

            diag = static[:, :, 0:25]
            diag = torch.FloatTensor(diag).to(device)
            batch_diag = diag[:, 0, :].to('cpu')

            edge_info = torch.where(torch.sum(batch_diag, 0) >= 3, torch.ones(batch_diag.shape[1]),
                                    torch.zeros(batch_diag.shape[1]))
            edge_index = torch.nonzero(edge_info == 1).squeeze().tolist()
            edge = [MakeEdge(batch_diag[:, edge_index[i]], i) for i in range(len(edge_index))]
            hyperedge_index = torch.hstack(edge).long().to(device)

            batch_input = torch.cat((batch_x, diag), 2)
            out_list, att = model(batch_input, hyperedge_index, sorted_length)

            batch_y = batch_y.long()
            y_out = batch_y.cpu().numpy()
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out),
                                                              y=y_out)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

            loss_ = [att[i] * criterion(out_list[i], batch_y) for i in range(len(out_list))]
            loss = sum(loss_)
            batch_loss.append(loss.cpu().detach().numpy())

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

        train_loss.append(np.mean(np.array(batch_loss)))
        # scheduler.step()

        with torch.no_grad():
            y_true = []
            y_pred = []
            model.eval()

            for step in range(valid_steps):
                ret = next(valid_gen)
                x = [np.hstack((ret[0][0][i], ret[0][1][i])) for i in range(len(ret[0][0]))]
                y = ret[1]
                static = ret[2]
                sorted_length = ret[3]
                batch_x = torch.FloatTensor(x).to(device)
                batch_y = torch.LongTensor(y).squeeze().to(device)

                x_mean = torch.stack(get_median(batch_x)).to(device)
                x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
                batch_x = torch.where(batch_x == -1, x_mean, batch_x)

                diag = static[:, :, 0:25]
                diag = torch.FloatTensor(diag).to(device)
                batch_diag = diag[:, 0, :].to('cpu')

                edge_info = torch.where(torch.sum(batch_diag, 0) >= 3, torch.ones(batch_diag.shape[1]),
                                        torch.zeros(batch_diag.shape[1]))
                edge_index = torch.nonzero(edge_info == 1).squeeze().tolist()
                edge = [MakeEdge(batch_diag[:, edge_index[i]], i) for i in range(len(edge_index))]
                hyperedge_index = torch.hstack(edge).long().to(device)

                batch_input = torch.cat((batch_x, diag), 2)
                out_list, att = model(batch_input, hyperedge_index, sorted_length)

                out = torch.stack(out_list)
                output = torch.median(out, 0).values
                output = F.softmax(output, dim=1)

                batch_y = batch_y.long()
                y_pred.append(output)
                y_true.append(batch_y)

            y_pred = torch.cat(y_pred, 0)
            y_true = torch.cat(y_true, 0)
            test_y_pred = y_pred.cpu().detach().numpy()
            test_y_true = y_true.cpu().detach().numpy()
            ret = print_metrics_binary(test_y_true, test_y_pred)

            cur_auroc = ret['auroc']

            if cur_auroc > max_auroc:
                max_auroc = cur_auroc
                best_epoch = each_epoch
                state = {
                    'net': model.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': each_epoch
                }
                torch.save(state, file_name)

    return best_epoch


def test(test_gen,
         test_steps,
         in_dim,
         hidden_dim,
         phi,
         gcn_dim,
         MLP_dims,
         N,
         drop_prob,
         seed,
         file_name,
         device):

    setup_seed(seed)
    model = Model(in_dim, hidden_dim, phi, gcn_dim, MLP_dims, N, drop_prob).to(device)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    y_true = []
    y_pred = []
    for step in range(test_steps):
        ret = next(test_gen)
        x = [np.hstack((ret[0][0][i], ret[0][1][i])) for i in range(len(ret[0][0]))]
        y = ret[1]
        static = ret[2]
        sorted_length = ret[3]
        batch_x = torch.FloatTensor(x).to(device)
        batch_y = torch.LongTensor(y).squeeze().to(device)

        x_mean = torch.stack(get_median(batch_x)).to(device)
        x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
        batch_x = torch.where(batch_x == -1, x_mean, batch_x)

        diag = static[:, :, 0:25]
        diag = torch.FloatTensor(diag).to(device)
        batch_diag = diag[:, 0, :].to('cpu')

        edge_info = torch.where(torch.sum(batch_diag, 0) >= 3, torch.ones(batch_diag.shape[1]),
                                torch.zeros(batch_diag.shape[1]))
        edge_index = torch.nonzero(edge_info == 1).squeeze().tolist()
        edge = [MakeEdge(batch_diag[:, edge_index[i]], i) for i in range(len(edge_index))]
        hyperedge_index = torch.hstack(edge).long().to(device)

        batch_input = torch.cat((batch_x, diag), 2)
        out_list, att = model(batch_input, hyperedge_index, sorted_length)

        out = torch.stack(out_list)
        output = torch.median(out, 0).values
        output = F.softmax(output, dim=1)

        batch_y = batch_y.long()
        y_pred.append(output)
        y_true.append(batch_y)

    y_pred = torch.cat(y_pred, 0)
    y_true = torch.cat(y_true, 0)
    test_y_pred = y_pred.cpu().detach().numpy()
    test_y_true = y_true.cpu().detach().numpy()
    ret = print_metrics_binary(test_y_true, test_y_pred)
    cur_auroc = ret['auroc']
    cur_auprc = ret['auprc']
    cur_acc = ret['acc']
    cur_precision = ret['prec1']
    cur_recall = ret['rec1']
    cur_f1 = ret['f1_score']
    cur_minpse = ret['minpse']

    results = {'AUROC': cur_auroc, 'AUPRC': cur_auprc, 'Accuracy':cur_acc, 'Precision':cur_precision, 'Recall':cur_recall, 'F1':cur_f1, 'Minpse':cur_minpse}

    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--gcn_dim", type=int)
    parser.add_argument("--MLP_dims", type=str)
    parser.add_argument("--N", type=int)
    parser.add_argument("--drop_prob", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--time_length", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_s_path", type=str)
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()

    in_dim = args.in_dim
    hidden_dim = args.hidden_dim
    phi = args.phi
    gcn_dim = args.gcn_dim
    MLP_dims = args.MLP_dims
    N = args.N
    drop_prob = args.drop_prob
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    time_length = args.time_length
    data_path = args.data_path
    data_s_path = args.data_s_path
    file_name = args.file_name

    train_gen, train_steps, valid_gen, valid_steps, test_gen, test_steps = data_process_eICU(data_path, data_s_path, time_length)
    best_epoch = train(train_gen, train_steps, valid_gen, valid_steps, in_dim, hidden_dim, phi, gcn_dim, MLP_dims, N, drop_prob, lr, seed, epochs, file_name, device)
    results = test(test_gen, test_steps, in_dim, hidden_dim, phi, gcn_dim, MLP_dims, N, drop_prob, seed, file_name, device)
    print(results)

