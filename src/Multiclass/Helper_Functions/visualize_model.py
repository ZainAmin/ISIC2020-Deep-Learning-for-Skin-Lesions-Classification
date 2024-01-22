def visualize_model(model, loader, device, num_images=6):
    """
    Visualizes the predictions of a PyTorch model on a subset of the given data loader.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to visualize.
    - loader (torch.utils.data.DataLoader): The data loader containing images and labels.
    - device (str): Device ('cuda' or 'cpu') on which the model and data are located.
    - num_images (int): Number of images to visualize. Default is 6.
    """
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                fig = plt.figure(figsize=(5, 5))
                imshow(inputs.cpu().data[j])  # You need to define the "imshow" function
                print(f'Actual: {labels[j]}, Predicted: {preds[j]}')
                plt.show()

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)