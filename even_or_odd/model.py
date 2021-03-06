import torch
import torch.nn as nn
from even_or_odd import data_reader
from tqdm import tqdm


class Config:
    emb_dim = 5
    hidden_dim = 5
    lstm_layers = 1
    bidirectional = False

    max_epoch = 60
    print_loss = 5000
    batch_size = 100


class Model(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.emb = nn.Embedding(11, config.emb_dim)
        self.lstm = nn.LSTM(input_size=config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.lstm_layers,
                            bidirectional=config.bidirectional, batch_first=True, bias=True)
        is_bi = 2 if config.bidirectional else 1
        self.linear = nn.Linear(config.hidden_dim * is_bi, 2)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x

    def predict(self, x):
        y = self(x)
        y = y.argmax(dim=-1)
        return y


def train():
    config = Config()
    model = Model(config)
    trainset = data_reader.TextDataSet()
    traindata = data_reader.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    testset = data_reader.TextDataSet('test')
    testdata = data_reader.DataLoader(testset, batch_size=config.batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    for epoch in range(config.max_epoch):
        for i, (x, gold) in enumerate(traindata):
            optim.zero_grad()
            y = model(x)
            loss = loss_fn(y, gold)
            loss.backward()
            optim.step()
            if i % config.print_loss == 0 or i == len(traindata) - 1:
                batch_size = x.size(0)
                acc = (gold == model.predict(x)).sum()
                print(f'epoch={epoch}  i={i}  loss={loss.item():0.4f}  训练集acc={acc.item() / batch_size}')

    total = 0
    right = 0
    model.eval()
    with torch.no_grad():
        for x, gold in tqdm(testdata, total=len(testdata)):
            total += x.size(0)
            right += (gold == model.predict(x)).sum().item()
    print(f'\nepoch={epoch}  测试集acc={right / total}')


if __name__ == '__main__':
    # 训练模型
    train()

    """
    使用二进制数据：
    epoch=10  测试集acc=0.876
    epoch=20  测试集acc=1.0

    使用文本数据：
    epoch=10  测试集acc=0.853
    epoch=30  测试集acc=1.0
    """
