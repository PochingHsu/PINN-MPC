import torch
import torch.nn as nn
import torch.nn.functional as F
import time

seed = 42
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = layer_dim
        self.output_dim = output_dim
        # self.dropout = dropout_prob
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleDict()
        self.layers["input"] = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        for i in range(self.n_layers):
            self.layers[f"hidden_{i}"] = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            # self.layers[f"dropout{i}"] = nn.Dropout(dropout_prob)
        self.layers["output"] = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.layers["input"](self.flatten(x))
        for i in range(self.n_layers):
            x = F.tanh(self.layers[f"hidden_{i}"](x))  # tanh/relu
            # x = self.layers[f"dropout{i}"](x)
        return self.layers["output"](x)


def get_model(model, model_params):
    models = {
        "ann": NeuralNetwork
    }
    return models.get(model.lower())(**model_params)


class LossFuc:
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def nnloss(self):
        yh = self.model(self.x_train)
        loss = torch.mean((yh - self.y_train) ** 2)  # MSE
        return loss

    def pinnloss(self, rvc, dt, ua, t_amb, hg, rc, vdot_air, tevap, lamda):
        yh = self.model(self.x_train)
        loss1 = torch.mean((yh - self.y_train) ** 2)  # MSE
        x_pinn = self.x_train.requires_grad_(True)
        yhp = self.model(x_pinn)
        # The residual of the differential equation
        physics = rvc * (yhp-x_pinn[:, 0])/dt - ua * (t_amb - yhp) - hg + rc * vdot_air * (x_pinn[:, 0] - tevap)
        loss2 = lamda * torch.mean(physics ** 2)
        return loss1 + loss2


class Optimization:
    def __init__(self, model, mo, optimizer):
        self.model = model
        self.mo = mo
        self.optimizer = optimizer

    def train(self, x_train, y_train, n_epochs):
        if self.mo == 'NN':
            start = time.time()
            print("start")
            for i in range(n_epochs):
                self.optimizer.zero_grad()
                nnlossfuc = LossFuc(model=self.model, x_train=x_train, y_train=y_train)
                loss = nnlossfuc.nnloss()
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{:10}/{}], Loss: {:.10f}'.format(i + 1, n_epochs, loss.item()))
            end = time.time()
            print(end - start)

        if self.mo == 'PINN':
            start = time.time()
            print("start")
            for i in range(n_epochs):
                dt = 60
                rvc = 1.29 * 1029 * 27
                ua = 2.35 * 2 * 3 * (3 + 3)
                rc = 1.29 * 1029
                hg = (73 + 59) * 5
                t_amb = x_train[:, 1]
                vdot_air = x_train[:, 3]
                tevap = x_train[:, 2]
                self.optimizer.zero_grad()
                lamda = 1e-9
                nnlossfuc = LossFuc(model=self.model, x_train=x_train, y_train=y_train)
                loss = nnlossfuc.pinnloss(rvc, dt, ua, t_amb, hg, rc, vdot_air, tevap, lamda)
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{:10}/{}], Loss: {:.10f}'.format(i + 1, n_epochs, loss.item()))
            end = time.time()
            print(end - start)
        loss_train = loss.detach()  # MSE train
        print('MSE of train: {:.10f}'.format(loss_train))
