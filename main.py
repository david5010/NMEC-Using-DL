import torch
import torch.nn as nn
import torch.optim as optim

def train(model, optimizer, criterion, train_loader, device):
    """
    Standard Training Loop
    Returns Training Error
    """
    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(train_loader)

def evaluate(model, criterion, test_loader, device):
    """
    Standard Evaluate Loop
    Returns Testing Error
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            total_loss += loss.item()

    return total_loss/len(test_loader)

def train_and_evaluate(model, optimizer, criterion,
                       train_loader, test_loader,
                       num_epochs, device):
    """
    Simplify Training and Evaluation
    """
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        test_loss = evaluate(model, criterion, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return train_losses, test_losses