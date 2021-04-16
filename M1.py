import torch
from torchinfo import summary
from mobilenet_rm_filt_pt import MobileNetv1, remove_channel
import torch.nn.utils.prune as prune


def channel_fraction_pruning(model, fraction):
    model.conv1 = prune.ln_structured(model.conv1, name="weight", amount=fraction, n=1, dim=0)

    for i in range (0,13):
        model.layers[i].conv2 = prune.ln_structured(model.layers[i].conv2, name="weight", amount=fraction, n=1, dim=0)



if __name__ == '__main__':

    # ########## Load Trained Model ##########
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Device: {device}")

    # model = MobileNetv1().to(device)

    # state_dict = torch.load('mbnv1_pt.pt', map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # model.to(device)

    # print("Model Beginning Summary")
    batch_size = 1
    # summary(model, input_size=(batch_size,3, 32, 32))


    # ########## Prune Model ##########
    # channel_fraction_pruning(model, 0.5)
    # summary(model,input_size=(batch_size, 3, 32, 32))

    # cleaned_model = remove_channel(model)
    # summary(cleaned_model, input_size=(batch_size,3, 32, 32))


    # ########## Finetune Model ##########
    # model = MobileNetv1().to(device)
    # state_dict = torch.load('mbnv1_pt.pt', map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # model.to(device)

    fractions_to_prune = [0.05, 0.25, 0.5, 0.75, 0.9]
    epochs_to_train = [0, 3, 5]

    for frac in fractions_to_prune:
        print(f"Testing Pruning Fraction: {frac}")
        for e in epochs_to_train:

            print(f"Epoch: {e}")

            model = MobileNetv1().to(device)
            state_dict = torch.load('mbnv1_pt.pt', map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.to(device)

            channel_fraction_pruning(model, frac)
            summary(model, input_size=(batch_size, 3, 32, 32), verbose=0)
            cleaned_model = remove_channel(model)

            # add the finetuning
            # Retrain?

            # save for later use
            torch.save(cleaned_model, f'pt_models/model_epoch_{e}_frac_{frac}.pt')
            torch.onnx.export(cleaned_model, torch.randn(1, 3, 32, 32), f'onnx_models/model_epoch_{e}_frac_{frac}.onnx', export_params=True, opset_version=10)