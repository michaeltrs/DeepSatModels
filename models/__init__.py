from models.UNet3D.unet3d import UNet3D
from models.UNet3D.unet3df import UNet3D_CSCL
from models.CropTypeMapping.models import FCN_CRNN
from models.BiConvRNN.biconv_rnn import BiRNNSequentialEncoder
from models.TSViT.TSViTdense import TSViT
from models.TSViT.TSViTcls import TSViTcls

def get_model(config, device):
    model_config = config['MODEL']

    if model_config['architecture'] == "UNET3Df":
        return UNet3D_CSCL(model_config).to(device)

    if model_config['architecture'] == "UNET3D":
        return UNet3D(model_config).to(device)

    if model_config['architecture'] == "UNET2D-CLSTM":  # "FCN_CRNN":
        return FCN_CRNN(model_config).cuda()

    if model_config['architecture'] == "ConvBiRNN":
        return BiRNNSequentialEncoder(model_config, device).to(device)

    if model_config['architecture'] == "TSViTcls":
        model_config['device'] = device
        return TSViTcls(model_config).to(device)

    if model_config['architecture'] == "TSViT":
        return TSViT(model_config).to(device)

    else:
        raise NameError("Model architecture %s not found, choose from: 'UNET3D', 'UNET3Df', 'UNET2D-CLSTM', 'TSViT', 'TSViTcls'")
