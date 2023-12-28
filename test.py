import pygame as pg
from pygame.locals import *
import torch as th
import torch.nn as nn
import torch.nn.functional as nf


class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)

        x = nf.relu(self.conv1(x))
        x = nf.max_pool2d(x, 2, 2)

        x = nf.relu(self.conv2(x))

        x = x.view(input_size, -1)

        x = nf.relu(self.fc1(x))

        x = self.fc2(x)

        output = nf.log_softmax(x, dim=1)

        return output


def update_data():
    global data_raw, data
    data = th.tensor(data_raw).float()
    data = th.reshape(data, (1, 28, 28))
    data.to(device)


def get_result():
    update_data()
    return model(data).max(1, keepdim=True)[1].item()


def transform_surf():
    global surf, data_raw
    b = 0
    for i in range(28):
        for j in range(28):
            a = 0
            for k in range(10):
                for l in range(10):
                    a += surf.get_at((i*10+k, j*10+l))[0]

            b = int(a/2550)
            if b == 10:
                b -= 1
            data_raw[j][i] = b


th.no_grad()

device = th.device("cpu")
model: Digit = th.load('model.pt')
model.to(device)
model.eval()

data_raw = [[0 for _ in range(28)]for _ in range(28)]
data = th.tensor(data_raw).float()
data = th.reshape(data, (1, 28, 28))

screen = pg.display.set_mode((280, 280))
surf = pg.surface.Surface((280, 280))
curse = pg.surface.Surface((21, 21))
curse.fill((255, 255, 255))
drawing = False

running = True
while running:
    screen.fill((0, 0, 0))

    for event in pg.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == MOUSEBUTTONUP:
            drawing = False
        elif event.type == KEYDOWN:
            if event.key == K_c:
                transform_surf()
                print(get_result())
                surf.fill((0, 0, 0))

    if drawing:
        surf.blit(curse, (pg.mouse.get_pos()[0]-10, pg.mouse.get_pos()[1]-10))

    screen.blit(surf, (0, 0))

    last_pos = pg.mouse.get_pos()

    pg.display.flip()

pg.quit()
