import torch
from models.model import XRayClassifier
from utils.data_loader import get_dataloaders
from utils.metrics import accuracy
import datetime

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now()
    print("Start:", current_time.strftime("%H:%M:%S"))

    # Parametry
    num_classes = 4
    batch_size = 32
    epochs = 20
    learning_rate = 0.001
    data_dir = './dataset/'

    # Dane i model
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size)
    model = XRayClassifier(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(outputs, labels)

        current_time = datetime.datetime.now()
        print("Godzina:", current_time.strftime("%H:%M:%S"))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, '
              f'Accuracy: {train_acc/len(train_loader):.4f}')

if __name__ == "__main__":
    train()
