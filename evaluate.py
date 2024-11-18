import torch
from models.cnn_model import XRayClassifier
from utils.data_loader import get_dataloaders
from utils.metrics import accuracy

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parametry
    num_classes = 4
    batch_size = 32
    data_dir = './data/'
    model_path = './model.pth'

    # Dane i model
    _, _, test_loader = get_dataloaders(data_dir, batch_size)
    model = XRayClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    # Ewaluacja
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_acc += accuracy(outputs, labels)

    print(f'Test Accuracy: {test_acc/len(test_loader):.4f}')

if __name__ == "__main__":
    evaluate()
