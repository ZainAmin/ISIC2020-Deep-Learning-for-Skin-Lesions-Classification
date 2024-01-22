def imshow(inp, title=None):
    """
    Display a PyTorch tensor as an image.

    Parameters:
    - inp (torch.Tensor): Input tensor representing an image.
    - title (str): Title of the displayed image.

    Note:
    - The function assumes that the input tensor is normalized using mean and std values.
    """
    inp = inp.numpy().transpose((1, 2, 0))

    # Denormalize
    mean = np.array([0.6138, 0.5056, 0.4985])
    std = np.array([0.1611, 0.1672, 0.1764])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
