import random
import json
import torch
from torch.utils.data import DataLoader, Dataset


def make_data(train_num, test_num, dim=32):
    # 生成数据，dim是数值位数
    max_value = 2 ** (dim - 1)
    total = train_num + test_num
    assert max_value > total
    print(f'max_value={max_value}')
    nums = set()
    while True:
        nums.add(random.randint(0, max_value))
        if len(nums) >= total:
            break
    nums = list(nums)
    random.shuffle(nums)
    with open('train.json', 'w') as f:
        f.write(json.dumps(nums[: train_num]))
    with open('test.json', 'w') as f:
        f.write(json.dumps(nums[train_num:]))


class BinDataSet(Dataset):
    # 以二进制文本表示数字
    def __init__(self, dtype='train', dim=32):
        if dtype == 'train':
            path = 'train.json'
        else:
            path = 'test.json'
        self.nums = json.load(open(path))
        print(f'dtype={dtype}  数据数量={len(self.nums)}')
        self.dim = dim

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, item):
        num = self.nums[item]
        num_str = bin(num)[2:]
        num_str = '0' * (self.dim - len(num_str)) + num_str
        num_list = [int(t) for t in num_str]
        if num % 2 == 0:
            gold = 0
        else:
            gold = 1
        return torch.tensor(num_list, dtype=torch.long), gold


class TextDataSet(Dataset):
    # 以文本来表示数字
    def __init__(self, dtype='train', dim=32):
        if dtype == 'train':
            path = 'train.json'
        else:
            path = 'test.json'
        self.nums = json.load(open(path))
        print(f'dtype={dtype}  数据数量={len(self.nums)}')
        max_value = 2 ** (dim - 1)
        self.dim = len(str(max_value))
        self.vocab = {}
        for i in range(10):
            self.vocab[str(i)] = len(self.vocab)
        self.vocab[' '] = len(self.vocab)

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, item):
        num = self.nums[item]
        num_str = str(num)
        num_str = ' ' * (self.dim - len(num_str)) + num_str
        num_list = [self.vocab[t] for t in num_str]
        if num % 2 == 0:
            gold = 0
        else:
            gold = 1
        return torch.tensor(num_list, dtype=torch.long), gold


if __name__ == '__main__':
    # 生成训练
    make_data(1000, 20000, dim=32)
