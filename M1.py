import torch
from torchinfo import summary
from mobilenet_rm_filt_pt import MobileNetv1, remove_channel
import torch.nn.utils.prune as prune


def channel_fraction_pruning(model, fraction):
    #print(list(model.named_parameters()))
    # you have to iterate over all the blocks and basically do if isinstance of xyz, do this, elif isinstance dwconv, continue...
    # weights = model.state_dict()
    # layers = list(model.state_dict())
    # print(layers)

    model.conv1 = prune.ln_structured(model.conv1, name="weight", amount=fraction, n=1, dim=0)

    for i in range (0,13):
        model.layers[i].conv2  = prune.ln_structured(model.layers[i].conv2, name="weight", amount=fraction, n=1, dim=0)

    return model


if __name__ == '__main__':

    ########## Load Trained Model ##########
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MobileNetv1().to(device)

    state_dict = torch.load('mbnv1_pt.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)

    print("Model Beginning Summary")
    batch_size = 1
    summary(model, input_size=(batch_size,3, 32, 32))

    print("--------------")
    ########## Prune Model ##########
    pruned_model = channel_fraction_pruning(model, 0.5)
    summary(pruned_model,input_size=(batch_size, 3, 32, 32))

    cleaned_model = remove_channel(pruned_model)
    summary(cleaned_model, input_size=(batch_size,3, 32, 32))


    ########## Finetune Model ##########



######## not working below
    fraction=[0.05,0.25,0.5,0.75,0.9]
    epochs=[0,3,5]

    for frac in fraction:
        for e in epochs:
            pruned_model = channel_fraction_pruning(model, frac)
            cleaned_model = remove_channel(pruned_model)

            # add the finetuning


            # save for later use
            torch.save(cleaned_model, f'model_frac_{frac}_epoch_{e}.pt')
   