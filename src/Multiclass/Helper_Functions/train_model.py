def train_model(model, dataloaders, criterion, optimizer, scheduler, checkpoint_path, early_stop_patience=10, num_epochs=25, writer_path='', load_checkpoint=None):
    """
    Trains a PyTorch model using the specified configurations.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be trained.
    - dataloaders (dict): A dictionary containing PyTorch data loaders for training and validation sets.
    - criterion: The loss function used for optimization.
    - optimizer: The optimization algorithm.
    - scheduler: Learning rate scheduler.
    - checkpoint_path (str): The path to save model checkpoints during training.
    - early_stop_patience (int): Number of epochs to tolerate without improvement before early stopping.
    - num_epochs (int): Number of training epochs.
    - writer_path (str): Path to store TensorBoard logs.
    - load_checkpoint (str): Optional path to load a pre-trained model checkpoint.

    Returns:
    - model (torch.nn.Module): The trained PyTorch model.
    """
    since = time.time()

    best_weights_path = checkpoint_path[:-4] + '_best.pth'

    writers = {
        'train': SummaryWriter(writer_path + '/log'),
        'val': SummaryWriter(writer_path + '/log_val')
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_kappa = 0.0
    early_stop_count = 0
    prev_epoch = -1

    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        model.load_state_dict(checkpoint['model_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        criterion = checkpoint['criterion']
        prev_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_kappa = checkpoint['best_kappa']
        early_stop_count = checkpoint['early_stop_count']

    for epoch in range(prev_epoch + 1, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_labels = []
            running_preds = []

            for batch in tqdm(dataloaders[phase]):
                inputs = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_labels = running_labels + labels.int().cpu().tolist()
                running_preds = running_preds + preds.int().cpu().tolist()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(running_labels, running_preds)
            epoch_kappa = cohen_kappa_score(running_labels, running_preds)

            if phase == 'val':
                scheduler.step(epoch_loss)
                lr_ = optimizer.param_groups[0]['lr']
                writers['train'].add_scalar('info/lr', lr_, epoch)

            print(f'{phase.capitalize()} Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Kappa: {epoch_kappa:.4f}')

            writers[phase].add_scalar('info/loss', epoch_loss, epoch)
            writers[phase].add_scalar('info/acc', epoch_acc, epoch)
            writers[phase].add_scalar('info/kappa', epoch_kappa, epoch)

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if epoch_kappa > best_kappa:
                    best_kappa = epoch_kappa
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, best_weights_path)

        torch.save({
            'model_state': model.state_dict(),
            'criterion': criterion,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
            'best_kappa': best_kappa,
            'early_stop_count': early_stop_count
        }, checkpoint_path)

        if early_stop_count > early_stop_patience:
            print(f'Early stop after {epoch + 1} epochs')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation loss: {best_loss:.4f}')
    print(f'Best validation Kappa: {best_kappa:.4f}')

    model.load_state_dict(best_model_wts)
    writers['train'].close()
    writers['val'].close()
    return model