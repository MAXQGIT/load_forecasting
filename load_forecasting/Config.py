import torch

class Config():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_path = 'ETT/ETTh2.csv'
        self.targer = 'OT'
        self.train_radio = 0.8
        self.test_radio = 0.9
        self.hidden = 64
        self.heads = 8
        self.main_size = 7
        self.date_size = 4
        self.epoch = 100
        self.lr = 0.001
        self.step_size = 10
        self.seq_len = 24
        self.batch_size = 64
        self.model_path = 'model/model.pt'
        self.output_dim = self.seq_len
