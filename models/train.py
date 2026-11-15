import torch
import torch.nn as nn
from models.model import Wide_and_Deep
from torch.utils.data import TensorDataset, DataLoader

def train(model, optimizer, criterion, train_loader, n_epochs,device='mps'):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.
        for X_batch_wide, X_batch_deep, y_batch in train_loader:
            X_batch_wide, y_batch, X_batch_deep = X_batch_wide.to(device), y_batch.to(device),X_batch_deep.to(device)
            y_pred = model(X_batch_wide,X_batch_deep)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.4f}")

def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean, device='mps'):
    model.eval()
    metrics = []
    with torch.no_grad():
        for X_batch_wide, X_batch_deep, y_batch in data_loader:
            X_batch_wide, y_batch, X_batch_deep = X_batch_wide.to(device), y_batch.to(device), X_batch_deep.to(device)
            y_pred = model(X_batch_wide, X_batch_deep)
            metric = metric_fn(y_pred, y_batch)
            metrics.append(metric)
    return aggregate_fn(torch.stack(metrics))