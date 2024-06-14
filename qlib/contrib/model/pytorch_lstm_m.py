# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from typing import Text, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.data.dataset.weight import Reweighter
from scipy.stats import spearmanr, pearsonr

class MLSTM(Model):
    """MLSTM Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        num_tasks=3,
        lambda_reg=0.01,
        seed=None,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("LSTM")
        self.logger.info("LSTM pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.num_tasks = num_tasks
        self.lambda_reg=lambda_reg

        self.logger.info(
            "LSTM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.lstm_model = MLSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_tasks=self.num_tasks,
            dropout=self.dropout,
        ).to(self.device)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.lstm_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.lstm_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")
    
    def pearson_correlation(self, x, y):
        # 确保张量是在同一设备上
        x = x.view(-1)
        y = y.view(-1)

        # 计算均值
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        # 计算差值
        diff_x = x - mean_x
        diff_y = y - mean_y

        # 计算差值的乘积之和
        sum_product = torch.sum(diff_x * diff_y)

        # 计算差值的平方和
        sum_square_x = torch.sum(diff_x ** 2)
        sum_square_y = torch.sum(diff_y ** 2)

        # 计算皮尔逊相关系数
        correlation = sum_product / torch.sqrt(sum_square_x * sum_square_y)

        return correlation


    def loss_fn(self, preds, labels, weights=None):
        # 假设preds是一个元组，包含所有任务的预测
        # 假设labels是一个元组，包含所有任务的真实标签
        # weights是一个元组，包含每个任务的权重，如果没有指定，则使用统一的权重
        num_tasks = len(preds)
        
        # 初始化总损失
        total_loss = 0.0
        
        # 计算每个子任务的损失
        for pred in preds:
            if weights is None:
                weight = torch.ones_like(labels)
            else:
                weight = weights[i]
            
            # 计算掩码，避免NaN值
            mask = ~torch.isnan(labels)
            weighted_pred = pred[mask] * weight[mask]
            true_label = labels[mask]
            
            ic = self.pearson_correlation(weighted_pred, true_label)
            task_loss = -ic  # 负号因为IC是正的，但损失需要最小化
            total_loss += task_loss
        
        # 计算因子间的相关性损失
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                corr = self.pearson_correlation(preds[i], preds[j])
                corr_loss = self.lambda_reg * corr
                total_loss += corr_loss
        
        return total_loss

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            m_pred = [x[mask] for x in pred]
            return -self.loss_fn(m_pred, label[mask], weights=None)

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.lstm_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.lstm_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores) 

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.lstm_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.lstm_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.lstm_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = [x.detach().cpu().numpy() for x in self.lstm_model(x_batch)]
                pred_t = np.array(pred).T
                preds.append(pred_t)
            
        columns = ["task_%d" % d for d in range(self.num_tasks)]
        all_pred = pd.DataFrame(np.vstack(preds), index=index, columns=columns)
        # 将平均值赋值给新列
        all_pred['score'] = all_pred.apply(lambda row: row.mean(), axis=1)
        print(all_pred)
        return all_pred


class MLSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, num_tasks=3, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # 为每个任务创建一个线性层
        self.fc_outs = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_tasks)])

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        # 前向传播LSTM
        out, _ = self.rnn(x)
        # 对每个任务使用不同的线性层
        task_outputs = [fc_out(out[:, -1, :]).squeeze() for fc_out in self.fc_outs]
        return task_outputs
