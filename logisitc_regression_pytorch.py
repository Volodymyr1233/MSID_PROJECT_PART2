import time
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from implemented_models.LogisticRegrPyTorch import LogisticRegrPyTorch
from utils import make_table


def main(device = "cpu"):
    x, y = make_classification(n_samples=1200000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, random_state=42)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    logistic_regr_pytorch = LogisticRegrPyTorch(input_dim=x_train.shape[1], device=device, lr=0.01)

    start_time = time.time()
    logistic_regr_pytorch.fit(x_train_t, y_train_t, epochs=10, batch_size=200000)
    duration = time.time() - start_time
    print(f"Czas treningu: {duration:.2f} sekund\n")

    train_preds, train_probs = logistic_regr_pytorch.predict(x_train_t)
    train_loss = round(log_loss(y_train_t.numpy(), train_probs), 4)
    train_acc = round(accuracy_score(y_train_t.numpy(), train_preds), 4)

    val_preds, val_probs = logistic_regr_pytorch.predict(x_val_t)
    val_loss = round(log_loss(y_val_t.numpy(), val_probs), 4)
    val_acc = round(accuracy_score(y_val_t.numpy(), val_preds), 4)

    test_preds, test_probs = logistic_regr_pytorch.predict(x_test_t)
    test_loss = round(log_loss(y_test_t.numpy(), test_probs), 4)
    test_acc = round(accuracy_score(y_test_t.numpy(), test_preds), 4)


    make_table(
        "Regresja logistyczna (PyTorch)",
        "Accuracy", "CE",
        train_acc, train_loss,
        val_acc, val_loss,
        test_acc, test_loss
    )


if __name__ == "__main__":
    device = 'cpu'
    print(f"Device: {device.upper()}")
    main(device=device)
