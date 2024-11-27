import os
import torch
from models.model import XRayClassifier
from utils.data_loader import get_dataloaders
from utils.metrics import accuracy
import datetime

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now()
    print("Start:", current_time.strftime("%H:%M:%S"))

    num_classes = 4
    batch_size = 32
    epochs = 20
    learning_rate = 0.001
    data_dir = './dataset/'

    #BADANIA
    num_layers = 2      #LICZBA WARSTW
    hidden_units = 256  #LICZBA NEURONÓW

    model_path = f'best_model-{num_layers}-{hidden_units}.pth'

    # 1 warstwa    32 64  92  128 192 256
    # 2 warstwy    32 64  92  128 192 256
    # 3 warstwy    32 64  92  128 192 256

    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size)


    model = XRayClassifier(num_classes, num_layers, hidden_units).to(device)


    if os.path.exists(model_path):
        print(f"Wczytywanie modelu z {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model wczytany")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

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

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += accuracy(outputs, labels)

        val_acc /= len(val_loader)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Zapisano nowy najlepszy model z {num_layers} warstwami, {hidden_units} neuronami"
                  f" z dokładnością walidacyjną: {best_val_accuracy:.4f}")

        current_time = datetime.datetime.now()
        print("Godzina:", current_time.strftime("%H:%M:%S"))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_acc/len(train_loader):.4f}, "
              f"Validation Loss: {val_loss/len(val_loader):.4f}, "
              f"Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    train()
