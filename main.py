
from modulation_classification import ModulationDataset, ModulationConvEncoder, calc_contrastive_loss
from modulation_classification_settings import EPOCHS, NUM_ITERS, DEVICE
import torch

if __name__ == '__main__':

    print('Signal Modulation Classification..')

    dataset = ModulationDataset(device=DEVICE)

    encoder = ModulationConvEncoder(dataset.num_classes).to(DEVICE)
    print(len(dataset))
    outX = encoder(dataset[0][0].unsqueeze(0))

    encoder.train()
    for epoch in range(EPOCHS):
        for iter in range(NUM_ITERS):

            two_samples = dataset.get_two_random_samples()
            sample1 = two_samples[0]
            sample2 = two_samples[1]

            y = torch.Tensor([1.0]).to(DEVICE)
            if sample1[1] == sample2[1]:
                y = torch.Tensor([0.0]).to(DEVICE) #same class samples

            enc1 = encoder(sample1[0].unsqueeze(0))
            enc2 = encoder(sample2[0].unsqueeze(0))

            loss = calc_contrastive_loss(enc1, enc2, y, 0)
            print(loss.item())


