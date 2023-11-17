import segmentation_models_pytorch as smp
model = smp.UnetPlusPlus(
    encoder_name="resnet50",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3     
)