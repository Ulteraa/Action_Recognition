import torch
import torchvision.datasets as dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import torch.optim as optim
import dataset_setup
import config
import torch
from tqdm import tqdm
class Action_Recognition_Model(nn.Module):
    def __init__(self, input_size, num_layer, num_hidden_size, class_number):
        super(Action_Recognition_Model, self).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.num_hidden_size = num_hidden_size
        self.class_number = class_number
        #self.rnn=nn.RNN(input_size=self.input_size,num_layers=self.num_layer,hidden_size=self.num_hiden_size,batch_first=True)
        #self.gru=nn.GRU(input_size=self.input_size,num_layers=self.num_layer,hidden_size=self.num_hiden_size,batch_first=True)
        #self.lstm = nn.LSTM(input_size=self.input_size, num_layers=self.num_layer, hidden_size=self.num_hiden_size,batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=self.num_layer, hidden_size=self.num_hidden_size,
                            batch_first=True, dropout=0.3)

        self.fc = nn.Linear(self.num_hidden_size, self.class_number)

    def forward(self, x):
        # h0=torch.zeros(self.num_layer,x.shape[0],self.num_hiden_size)
        # c0=torch.zeros(self.num_layer,x.shape[0],self.num_hiden_size)
        h0 = torch.zeros(self.num_layer, x.shape[0], self.num_hidden_size)
        c0 = torch.zeros(self.num_layer, x.shape[0], self.num_hidden_size)
        #out,_=self.rnn(x,h0)
        #out, _ = self.gru(x, h0)
        out, _ = self.lstm(x, (h0, c0))
        # out=out.reshape(out.shape[0],-1)
        # print(out.shape)
        out = self.fc(out[:, -1, :])
        return out


def train(data, device, optimizer, model, loss_fn):
    loop = tqdm(data)
    for _, (x, y) in enumerate(loop):
        x = x.to(config.device); y = y.to(config.device)
        predict_ = model(x)
        optimizer.zero_grad()
        loss = loss_fn(predict_, y)
        loss.backward()
        optimizer.step()
    return loss

def train_fn():
    training_data = dataset_setup.Dataset_(train=True, transform=True, n_frame=100)
    training_loader = DataLoader(training_data, batch_size=config.bach_size, shuffle=True)
    # test_data = dataset.MNIST(root='dataset/', train=False, download=True, transform=transform.ToTensor())
    # test_loader = DataLoader(test_data, batch_size=bach_size, shuffle=True)
    model = Action_Recognition_Model(config.input_size, config.num_layer,
                                     config.num_hidden_size, config.class_number).to(device=config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    print("Starting training...\n")
    losses = []
    for epoch in range(100):
        loss = train(training_loader, config.device, optimizer, model, loss_fn)
        losses.append(loss)
        print(f"Epoch {epoch} | Train Loss {loss}")
        acc = test(model, epoch)
    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]


def test(model, epoch):
    test_dataset = dataset_setup.Dataset_(train=False, transform=True, n_frame=100)
    test_data = DataLoader(test_dataset, batch_size=config.bach_size, shuffle=True)
    ns = 0; correct = 0
    with torch.no_grad():
        loop = tqdm(test_data)
        for _, (x, y) in enumerate(loop):
            x = x.to(config.device);
            y = y.to(config.device)
            ns += x.shape[0]
            predict_ = model(x)
            prediction = predict_ .argmax(dim=1)
            correct += (prediction == y).sum()
        test_acc = (correct / ns)
        print(f"test accuracy in {epoch} is  {test_acc }")
        return test_acc

# def plt2arr(fig):
#     rgb_str = fig.canvas.tostring_rgb()
#     (w,h) = fig.canvas.get_width_height()
#     rgba_arr = np.fromstring(rgb_str, dtype=np.uint8, sep='').reshape((w,h,-1))
#     return rgba_arr


# def visualize(out, color):
#     fig = plt.figure(figsize=(5, 5), frameon=False)
#     z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
#     plt.scatter(z[:, 0],
#                 z[:, 1],
#                 s=10,
#                 c=color.detach().cpu().numpy(),
#                 cmap="Set2"
#                 )
#     plt.savefig('tsne.png')
#     # fig.canvas.draw()

if __name__ == '__main__':
    train_fn()

# for epoch in range(epochs):
#     for _, (data, target) in enumerate(training_loader):
#         data = data.squeeze(1)
#         data = data.to(device=device)
#         target = target.to(device=device)
#         optimizer.zero_grad()
#         prediction = model(data)
#         loss = Criterion(prediction, target)
#         loss.backward()
#         optimizer.step()
# def accuracy(model,data):
#     num_correct=0
#     num_sample=0
#     for _, (data,target) in enumerate(data):
#         data = data.squeeze(1)
#         data = data.to(device=device)
#         target = target.to(device=device)
#         source=model(data)
#         _, prediction_ = source.max(1)
#         num_correct+= (prediction_==target).sum()
#         num_sample+=prediction_.shape[0]
#     print(f'got {num_correct}/{num_sample} with accutacy of {float(num_correct/num_sample)*100:2f}')
#
# accuracy(model,training_loader)
# accuracy(model,test_loader)