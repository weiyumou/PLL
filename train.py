def train_pll(device, model, dataloaders):
    model = model.to(device)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        for batch in dataloaders[phase]:
            batch = batch.permute(1, 0)  # (S, N)
            outputs = model(batch)  # (N, S)
            break
        break
