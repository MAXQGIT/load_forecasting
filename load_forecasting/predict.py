import pandas as pd
import torch
from Config import Config
import matplotlib.pyplot as plt

def picture(true,pre,date):
    plt.figure()
    plt.plot(true,label = 't')
    plt.plot(pre,label='p')
    plt.legend()
    plt.savefig('result/{}.png'.format(date))
    plt.clf()
    plt.close()


def read_data(cfg):
    data = pd.read_csv(cfg.data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] <= '2018-06-24') & (data['date'] >= '2018-06-10')].iloc[:-1, :]
    OT = list(data['OT'])
    data = data.drop('OT', axis=1)
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] =data['date'].dt.hour
    data['date'] = data['date'].dt.date.apply(lambda x: str(x))
    data['OT'] = OT
    return data


def predict(cfg):
    model = torch.load(cfg.model_path)
    data = read_data(cfg)
    date_list = pd.date_range('2018-06-10', '2018-06-23', freq='D').strftime('%Y-%m-%d')
    for i in range(len(date_list) - 1):
        print(date_list[i])
        day_data = data[data['date'] == date_list[i]]
        day_data = day_data.drop('date', axis=1)
        x = day_data.values
        x = torch.tensor(x)
        x = x.reshape(1, x.shape[0], x.shape[1]).to('cuda')
        pre = model(x.float())
        pre = pre.cpu().tolist()
        t_data = data[data['date'] == date_list[i+1]]
        print(date_list[i+1])
        print('~~'*50)
        ture = list(t_data['OT'])
        picture(ture,pre[0],date_list[i+1])


if __name__ == '__main__':
    cfg = Config()
    predict(cfg)
