import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

# Создаём нейронную сеть для двухклассовой классификации
# Структура аналогична NNet из лабораторной работы
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Tanh(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Tanh()
                                    )
    # прямой проход    
    def forward(self, X):
        pred = self.layers(X)
        return pred


# Загружаем данные
df = pd.read_csv('dataset_simple.csv')

# X - признаки: возраст (age) и доход (income)
X = torch.Tensor(df.iloc[:, 0:2].values)

# Нормализация признаков
X = (X - X.mean(dim=0)) / X.std(dim=0)

y = torch.Tensor(df.iloc[:, -1].values)


# Параметры сети
inputSize = X.shape[1]   # 2 признака: возраст и доход
hiddenSizes = 10          # число нейронов скрытого слоя
outputSize = 1            # 1 выходной нейрон (двухклассовая задача)

# Создаем экземпляр сети
net = NNet(inputSize, hiddenSizes, outputSize)

# Выводим веса сети
for name, param in net.named_parameters():
    print(name, param)


# Посчитаем ошибку необученной сети
with torch.no_grad():
    pred = net.forward(X)

err = sum(abs(y - pred)) / 2
print('\nОшибка до обучения (количество несовпавших ответов):')
print(err)


# Функция вычисления ошибки
lossFn = nn.MSELoss()

# Оптимизатор
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Обучение
epohs = 500
for i in range(0, epohs):
    pred = net.forward(X)       # прямой проход - делаем предсказания
    loss = lossFn(pred, y)      # считаем ошибку
    optimizer.zero_grad()       # обнуляем градиенты
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())


# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

err = sum(abs(y - pred)) / 2
print('\nОшибка после обучения (количество несовпавших ответов):')
print(err)