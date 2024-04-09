import models
import os, utils, data


class configs(object):

    # hardware
    device = 'cuda:0'                  # device: cpu, cuda:0, cuda:1 and so on
    manual_seed = 1

    #path
    datapath = 'E:\\dataset\\fastMRI_brain\\Motion_Artifact_Reduction\\size=128x128_gap=9_in_plane'
    inference_path = 'E:\\dataset\\fastMRI_brain\\Motion_Artifact_Reduction\\size=128x128_gap=9_in_plnae\\Inference'
    snap_path = 'results\\fastMRI_brain\\UNAEN-gap=9_in_plane'

    # train
    load_weight = False
    batch_size = 4                      # 4 for CycleGAN, 16 for ISCL, 64 for UIDnet, 4 for UNAEN
    epochs = 50
    log_step = 1000                      # 250 for ISCL, 80 for UIDnet, 1000 for CycleGAN and UNAEN

    train_model = 'DR_CycleGAN'

    def __init__(self):

        self.inference_path += '\\{}'.format(self.train_model)

        if self.train_model == 'ISCL':
            self.batch_size = 16
            self.log_step = 250
        elif self.train_model == 'UIDnet':
            self.batch_size = 64
            self.log_step = 80

        if not os.path.exists(self.snap_path):
            os.makedirs(self.snap_path)



if __name__ == '__main__':

    config = configs()
    print('using device: %s.' % config.device)
    if config.manual_seed is not None:
        utils.init_random_seed(config.manual_seed)


    if config.train_model == 'UNAEN':
        # Our proposed method
        model = models.UNAEN(config)
    elif config.train_model == 'UNAEN_Gr':
        # Ablation: explicit one without Gr
        model = models.UNAEN_Gr(config)
    elif config.train_model == 'UNAEN_implicit':
        # Ablation: implicit one with Gr
        model = models.UNAEN_implicit(config)
    elif config.train_model == 'UNAEN_implicit_Gr':
        # Ablation: implicit one without Gr
        model = models.UNAEN_implicit_Gr(config)
    elif config.train_model == 'CycleGAN':
        # baseline method CycleGAN
        model = models.CycleGAN(config)
    elif config.train_model == 'ISCL':
        # baseline method ISCL, see TMI 2021 paper
        # "ISCL: Interdependent self-cooperative learning for unpaired image denoising" for more details.
        model = models.ISCL(config)
    elif config.train_model == 'UIDnet':
        # baseline method UIDNet, see AAAI 2020 paper
        # "End-to-end unparied image denoising with conditional adversarial networks" for more details.
        model = models.UIDnet(config)
    elif config.train_model == 'DR_CycleGAN':
        # baseline method DR_CycleGAN, see bioengineering 2023 paper
        # "Correction of Arterial-Phase Motion Artifacts in Gadoxetic Acid-Enhanced Liver MRI Using an Innovative Unsupervised Network" for more details.
        model = models.DR_CycleGAN(config)
    elif config.train_model =='UDDN':
        # baseline method UDDN, see CIBM 2023 paper
        # "Unsupervised dual-domain disentangled network for removal of rigid motion artifacts in MRI" for more details.
        model = models.UDDN(config)
    else:
        raise ValueError

    data_loader = data.get_dataloader(config)
    model.train(data_loader)

    test_loader = data.get_dataloader(config, evaluation=True)
    model.eval(test_loader)