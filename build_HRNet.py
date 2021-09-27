from .models.hrnet import HRNet
import torch


def create_hrnet(args, logger):
    model = HRNet(c=args.hrnet_channel_size,
    nof_joints=args.hrnet_nof_joints)

    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    checkpoint = torch.load(args.hrnet_model_weight, map_location=device)

    logger.info("=> Loading HRNet model from {}.".format(args.hrnet_model_weight))
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    
    logger.info("=> HRNet model is ready to use.")
    
    return model