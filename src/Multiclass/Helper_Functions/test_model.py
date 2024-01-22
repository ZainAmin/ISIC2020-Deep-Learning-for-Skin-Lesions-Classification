def test_model(model, loader, device, criterion):
    """
    Evaluates a PyTorch model on the specified data loader.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - loader (torch.utils.data.DataLoader): The data loader containing images and labels.
    - device (str): Device ('cuda' or 'cpu') on which the model and data are located.
    - criterion: The loss function used for evaluation.

    Returns:
    - test_loss (float): Loss on the test set.
    - test_acc (float): Accuracy on the test set.
    - test_kappa (float): Cohen's Kappa score on the test set.
    """
    since = time.time()

    model.eval()

    running_loss = 0.0
    running_corrects = 0
    running_labels = []
    running_preds = []

    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_labels = running_labels + labels.int().cpu().tolist()
            running_preds = running_preds + preds.int().cpu().tolist()

        test_loss = running_loss / len(loader.dataset)
        test_kappa = cohen_kappa_score(running_labels, running_preds)
        test_acc = accuracy_score(running_labels, running_preds)

        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Kappa: {test_kappa:.4f}')

        time_elapsed = time.time() - since
        print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        model.train()

        return test_loss, test_acc, test_kappa