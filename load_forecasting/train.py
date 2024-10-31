import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from Config import Config
from self_model import MAIN_MODEL
from dateset import TimeSeriesDataset


def read_data(cfg):
    data = pd.read_csv(cfg.data_path)
    OT = list(data['OT'])
    data = data.drop('OT', axis=1)
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour
    data = data.drop('date', axis=1)
    data['OT'] = OT
    return data

def tensor_data(cfg):
    data = read_data(cfg)
    dataset = TimeSeriesDataset(data.values, cfg.seq_len)
    total_size = len(dataset)
    train_size = int(total_size * cfg.train_radio)
    val_size = int(total_size * cfg.val_radio)
    test_size = total_size - train_size - val_size
    train_data, test_data, val_data = random_split(dataset, [train_size, val_size, test_size])
    train_data = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_data = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True)
    return train_data, val_data, test_data


def test(cfg, loss, test_data):
    model = torch.load(cfg.model_path, weights_only=False)
    test_loss_list = []
    for x, y in test_data:
        x, y = x.to(cfg.device), y.to(cfg.device)
        pre = model(x.float())
        test_loss = loss(y.float(), pre)
        test_loss_list.append(test_loss.item())
    return sum(test_loss_list) / len(test_loss_list)


def train(cfg):
    model = MAIN_MODEL(cfg).to(cfg.device)
    train_data, test_data, val_data = tensor_data(cfg)
    loss = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), cfg.lr)
    scheduler = StepLR(optim, step_size=cfg.step_size, gamma=0.5)
    for epoch in range(cfg.epoch):
        model.train()
        train_loss_list = []
        for x, y in train_data:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pre = model(x.float())
            optim.zero_grad()
            train_loss = loss(y.float(), pre)
            train_loss_list.append(train_loss.item())
            train_loss.backward()
            optim.step()
        scheduler.step()
        torch.save(model, cfg.model_path)
        train_loss1 = sum(train_loss_list) / len(train_loss_list)
        model.eval()
        eval_loss = test(cfg, loss, val_data)
        test_loss = test(cfg, loss, test_data)

        print('epoch:{} train_loss:{:.2f} val_loss:{:.2f}  test_loss:{:.2f}'.format(
            epoch + 1, train_loss1, eval_loss, test_loss))


if __name__ == '__main__':
    cfg = Config()
    train(cfg)
