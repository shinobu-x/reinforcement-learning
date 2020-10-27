def get_batch(x, y, batch_size, shuffle = True):
    if shuffle:
        x = x[torch.randperm(len(x))]
        y = y[torch.randperm(len(y))]
    batch = []
    for i in range(len(n) // batch_size):
        x_i = x[i * batch_size: (i + 1) * batch_size]
        y_i = y[i * batch_size: (i + 1) * batch_size]
        batch.append((x_i, y_i))
    return batch
