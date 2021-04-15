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






    #         Conv2d-1           [-1, 32, 32, 32]             864
    #    BatchNorm2d-2           [-1, 32, 32, 32]              64
    #         Conv2d-3           [-1, 32, 32, 32]             288
    #    BatchNorm2d-4           [-1, 32, 32, 32]              64
    #         Conv2d-5           [-1, 64, 32, 32]           2,048
    #    BatchNorm2d-6           [-1, 64, 32, 32]             128
    #          Block-7           [-1, 64, 32, 32]               0
    #         Conv2d-8           [-1, 64, 16, 16]             576
    #    BatchNorm2d-9           [-1, 64, 16, 16]             128
    #        Conv2d-10          [-1, 128, 16, 16]           8,192
    #   BatchNorm2d-11          [-1, 128, 16, 16]             256
    #         Block-12          [-1, 128, 16, 16]               0
    #        Conv2d-13          [-1, 128, 16, 16]           1,152
    #   BatchNorm2d-14          [-1, 128, 16, 16]             256
    #        Conv2d-15          [-1, 128, 16, 16]          16,384
    #   BatchNorm2d-16          [-1, 128, 16, 16]             256
    #         Block-17          [-1, 128, 16, 16]               0
    #        Conv2d-18            [-1, 128, 8, 8]           1,152
    #   BatchNorm2d-19            [-1, 128, 8, 8]             256
    #        Conv2d-20            [-1, 256, 8, 8]          32,768
    #   BatchNorm2d-21            [-1, 256, 8, 8]             512
    #         Block-22            [-1, 256, 8, 8]               0
    #        Conv2d-23            [-1, 256, 8, 8]           2,304
    #   BatchNorm2d-24            [-1, 256, 8, 8]             512
    #        Conv2d-25            [-1, 256, 8, 8]          65,536
    #   BatchNorm2d-26            [-1, 256, 8, 8]             512
    #         Block-27            [-1, 256, 8, 8]               0
    #        Conv2d-28            [-1, 256, 4, 4]           2,304
    #   BatchNorm2d-29            [-1, 256, 4, 4]             512
    #        Conv2d-30            [-1, 512, 4, 4]         131,072
    #   BatchNorm2d-31            [-1, 512, 4, 4]           1,024
    #         Block-32            [-1, 512, 4, 4]               0
    #        Conv2d-33            [-1, 512, 4, 4]           4,608
    #   BatchNorm2d-34            [-1, 512, 4, 4]           1,024
    #        Conv2d-35            [-1, 512, 4, 4]         262,144
    #   BatchNorm2d-36            [-1, 512, 4, 4]           1,024
    #         Block-37            [-1, 512, 4, 4]               0
    #        Conv2d-38            [-1, 512, 4, 4]           4,608
    #   BatchNorm2d-39            [-1, 512, 4, 4]           1,024
    #        Conv2d-40            [-1, 512, 4, 4]         262,144
    #   BatchNorm2d-41            [-1, 512, 4, 4]           1,024
    #         Block-42            [-1, 512, 4, 4]               0
    #        Conv2d-43            [-1, 512, 4, 4]           4,608
    #   BatchNorm2d-44            [-1, 512, 4, 4]           1,024
    #        Conv2d-45            [-1, 512, 4, 4]         262,144
    #   BatchNorm2d-46            [-1, 512, 4, 4]           1,024
    #         Block-47            [-1, 512, 4, 4]               0
    #        Conv2d-48            [-1, 512, 4, 4]           4,608
    #   BatchNorm2d-49            [-1, 512, 4, 4]           1,024
    #        Conv2d-50            [-1, 512, 4, 4]         262,144
    #   BatchNorm2d-51            [-1, 512, 4, 4]           1,024
    #         Block-52            [-1, 512, 4, 4]               0
    #        Conv2d-53            [-1, 512, 4, 4]           4,608
    #   BatchNorm2d-54            [-1, 512, 4, 4]           1,024
    #        Conv2d-55            [-1, 512, 4, 4]         262,144
    #   BatchNorm2d-56            [-1, 512, 4, 4]           1,024
    #         Block-57            [-1, 512, 4, 4]               0
    #        Conv2d-58            [-1, 512, 2, 2]           4,608
    #   BatchNorm2d-59            [-1, 512, 2, 2]           1,024
    #        Conv2d-60           [-1, 1024, 2, 2]         524,288
    #   BatchNorm2d-61           [-1, 1024, 2, 2]           2,048
    #         Block-62           [-1, 1024, 2, 2]               0
    #        Conv2d-63           [-1, 1024, 2, 2]           9,216
    #   BatchNorm2d-64           [-1, 1024, 2, 2]           2,048
    #        Conv2d-65           [-1, 1024, 2, 2]       1,048,576
    #   BatchNorm2d-66           [-1, 1024, 2, 2]           2,048
    #         Block-67           [-1, 1024, 2, 2]               0
    #        Linear-68                   [-1, 10]          10,250