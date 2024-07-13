import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, scl_loss
from data.augmentor import GraphAugmentor


class SGCL(GraphRecommender):
    def __init__(self, conf, training_set,  test_set):
        super(SGCL, self).__init__(conf, training_set, test_set)
        self.best_user_emb = None
        self.best_item_emb = None
        args = OptionConf(self.config['SGCL'])
        self.cl_rate = float(args['-beta'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        num_negative = int(self.config['num.negative'])
        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, num_negative, aug_type)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size, self.num_negative)):
                user_idx, pos_idx, neg_idx = batch
                user_emb, pos_item_emb, neg_item_emb = model.negative_mixup(user_idx, pos_idx, neg_idx)
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                scl_loss_ = self.cl_rate * model.cal_scl_loss([user_idx, pos_idx, neg_idx], dropped_adj1, dropped_adj2)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,
                                                    neg_item_emb) + scl_loss_
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch >= 5:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)

        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, num_negative, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.num_negative = num_negative
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type == 0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def forward_mix(self):
        self.dropout = nn.Dropout(0.1)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        user_embs = [self.embedding_dict['user_emb']]
        item_embs = [self.embedding_dict['item_emb']]
        # adj = self._sparse_dropout(self.sparse_norm_adj, 0.5)
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            ego_embeddings = self.dropout(ego_embeddings)
            user_embs.append(ego_embeddings[:self.data.user_num])
            item_embs.append(ego_embeddings[self.data.user_num:])
        user_embs = torch.stack(user_embs, dim=1)
        user_embs = torch.mean(user_embs, dim=1)
        return user_embs, item_embs

    def negative_mixup(self, user, pos_item, neg_item):
        user_emb, item_emb = self.forward_mix()
        u_emb = user_emb[user]
        negs = []
        for k in range(self.n_layers + 1):
            neg_emb = item_emb[k][neg_item]
            pos_emb = item_emb[k][pos_item]
            neg_emb = neg_emb.reshape(-1, self.num_negative, self.emb_size)
            alpha = torch.rand_like(neg_emb).cuda()
            neg_emb = alpha * pos_emb.unsqueeze(dim=1) + (1 - alpha) * neg_emb
            scores = (u_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
            indices = torch.max(scores, dim=1)[1].detach()
            chosen_neg_emb = neg_emb[torch.arange(neg_emb.size(0)), indices]
            negs.append(chosen_neg_emb)
        item_emb = torch.stack(item_emb, dim=1)
        item_emb = torch.mean(item_emb, dim=1)
        negs = torch.stack(negs, dim=1)
        negs = torch.mean(negs, dim=1)
        return u_emb, item_emb[pos_item], negs

    def cal_scl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)
        return scl_loss(view1, view2, 0.1, 0.01, self.temp)