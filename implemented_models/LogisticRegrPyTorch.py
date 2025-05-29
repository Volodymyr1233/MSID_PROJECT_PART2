import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegrPyTorch:
    def __init__(self, input_dim, device='cpu', lr=0.01):
        self.device = torch.device(device)
        self.model = nn.Linear(input_dim, 1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def forward(self, x):
        x = x.to(self.device)
        logits = self.model(x)
        return logits

    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze(1)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()



            print(f"Epoch {epoch+1}/{epochs}")

    def predict(self, X, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.sigmoid(logits).squeeze(1)
            return (probs >= threshold).cpu().numpy(), probs.cpu().numpy()
