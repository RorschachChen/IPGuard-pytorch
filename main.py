import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch import optim

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='/data/cym/code/fingerprint/SAC/data', train=True, download=True,
                                             transform=transform_train)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

net = ResNet18()
net.load_state_dict(torch.load('checkpoint/model_last.th'))
net = net.to(device)
net = torch.nn.DataParallel(net)
net.eval()
cudnn.benchmark = True

iters = 1000
n = 100
k = 15
learning_rate = 0.001
epsilon = 8.0 / 255.0
alpha = 0.00784
final_samples = []
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(batch_idx)
    if len(final_samples) == n:
        break
    inputs = inputs.to(device)
    adv = inputs.detach()
    adv.requires_grad_()
    output = net(inputs)
    i = output.data.max(1)[1]
    j = output.data.min(1)[1]
    optimizer = optim.Adam([adv], lr=learning_rate, weight_decay=0.0002)
    for _ in range(iters):
        optimizer.zero_grad()
        logits = net(adv)
        t = None
        for c in range(10):
            if c != i and c != j:
                if t is None:
                    t = c
                else:
                    if logits[0][t] < logits[0][c]:
                        t = c
        loss = F.relu(logits[0][i] - logits[0][j] + k) + F.relu(logits[0][t]- logits[0][i])
        if loss <= 1e-5:
            final_samples.append(adv.detach().cpu())
            print('find one!')
            break
        loss.backward()
        optimizer.step()
    # print(loss)
torch.save(final_samples, 'query.pth')