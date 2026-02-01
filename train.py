import argparse
import time
from torch.utils.data import DataLoader
from dataset import *
from model.baseline import *
from model.DWCANet import *

from metrics import *
from loss import *
import shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")

parser.add_argument("--model_names", default=['DWCANet'], nargs='+',
                    help="model_name: 'baseline','DWCANet'")
parser.add_argument("--dataset_names", default=[ 'IRSTD-1K'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K',")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--resume", default=None, nargs='+', help="Resume from exisiting checkpoints (default: None or ['./xxx.pth.tar'])")

parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default='./logs', type=str, help="Save path of checkpoints")
parser.add_argument("--pretrained", default=None, nargs='+', help="Load pretrained checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")
parser.add_argument("--optimizer_name", default='AdamW', type=str, help="optimizer name: Adam, Adagrad,AdamW,SGD")
parser.add_argument("--optimizer_settings", default={'lr': 1e-3, 'weight_decay': 1e-2}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='CosineAnnealingLR', type=str, help="scheduler name: CosineAnnealingLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict,
                    help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--intervals", type=int, default=5, help="Intervals for print epoch")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")

parser.add_argument("--begin_test", default=200, type=int, help="From which round should we start testing")
parser.add_argument("--every_print", default=10, type=int, help="How many rounds do you print detailed test information every time")
parser.add_argument("--every_save", default=10, type=int, help="How many rounds do we save the regular weights every time")

global opt
opt = parser.parse_args()

## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
    opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)
timestamp = time.strftime("%Y%m%d_%H%M")

def train():
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    if opt.model_name == 'baseline':
        net = Baseline().cuda()
    elif opt.model_name == 'DWCANet':
        net = DWCANet().cuda()

    best_mIoU = 0.0
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    net.train()
    # Resume Logic
    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth, weights_only=False)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                best_mIoU = ckpt.get('best_mIoU', 0.0)
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']

    # Pretrained Logic
    if opt.pretrained:
        for pretrained_pth in opt.pretrained:
            if opt.dataset_name in pretrained_pth and opt.model_name in pretrained_pth:
                print(f"Loading pre-trained weights: {pretrained_pth}")
                ckpt = torch.load(pretrained_pth, weights_only=False)
                model_dict = net.state_dict()
                pretrained_dict = {k: v for k, v in ckpt['state_dict'].items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                print(f"Successfully loaded layers: {len(pretrained_dict)} / {len(model_dict)}")
                model_dict.update(pretrained_dict)
                net.load_state_dict(model_dict)


    net = torch.nn.DataParallel(net)
    criterion = SoftIoULoss()


    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        step1 = int(opt.nEpochs * 0.5)
        step2 = int(opt.nEpochs * 0.75)
        opt.scheduler_settings = {'epochs': opt.nEpochs, 'step': [step1, step2], 'gamma': 0.1}


    elif opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.nEpochs, 'min_lr': 1e-5}


    elif opt.optimizer_name == 'AdamW':
        opt.optimizer_settings = {'lr': 1e-3, 'weight_decay': 1e-2}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.nEpochs, 'min_lr': 1e-6}
    else:
        opt.nEpochs = opt.scheduler_settings['epochs']

    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                             opt.scheduler_settings)

    opt.f.write(f"total epoch: {opt.nEpochs}\n"
                f"optimizer: {opt.optimizer_name}\n"
                f"Best mIoU : {best_mIoU:.4f}\n")
    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = img.cuda(), gt_mask.cuda()

            gt_mask[gt_mask > 0.5] = 1.0
            gt_mask[gt_mask <= 0.5] = 0.0

            if img.shape[0] == 1:
                continue
            pred = net(img)
            loss = criterion(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (idx_epoch + 1) % opt.every_print == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, Train Loss---%f, lr---%f'
                  % (idx_epoch + 1, total_loss_list[-1], scheduler.get_last_lr()[0]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []

        if (idx_epoch + 1) % opt.every_save == 0 or (idx_epoch + 1) == opt.nEpochs:
            regular_save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
                'total_loss': total_loss_list,
                'best_mIoU': best_mIoU,
            }, regular_save_pth)

        if (idx_epoch + 1) >= opt.begin_test:
            temp_save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_temp.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
                'total_loss': total_loss_list,
                'best_mIoU': best_mIoU,
            }, temp_save_pth)

            is_print_epoch = ((idx_epoch + 1) % opt.every_print == 0)
            if is_print_epoch:
                print(f"--- Epoch {idx_epoch + 1} Testing ---")

            current_mIoU = test(temp_save_pth, verbose=is_print_epoch)

            if current_mIoU > best_mIoU:
                best_mIoU = current_mIoU

                opt.f.write(f"New Best mIoU: {best_mIoU:.4f} at Epoch {idx_epoch + 1}\n")

                best_save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + str(idx_epoch + 1) +'_best.pth.tar'
                shutil.copyfile(temp_save_pth, best_save_pth)


def test(save_pth, verbose=True):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    if opt.model_name == 'baseline':
        net = Baseline().cuda()
    elif opt.model_name == 'DWCANet':
        net = DWCANet().cuda()


    try:
        ckpt = torch.load(save_pth, weights_only=False)
        net.load_state_dict(ckpt['state_dict'])
    except Exception as e:
        print(f"Weight loading failed: {e}")
        return 0.0

    net.eval()

    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()

    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
            img = img.cuda()
            pred = net(img)
            if isinstance(pred, list):
                pred = pred[-1]

            if (pred < 0).any() or (pred > 1).any():
                pred = torch.sigmoid(pred)
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)

    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()

    if verbose:
        print(f"--- Epoch Evaluation ({opt.model_name}) ---")
        print("pixAcc, mIoU:\t" + str(results1))
        print("PD, FA:\t" + str(results2))
        opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
        opt.f.write("PD, FA:\t" + str(results2) + '\n')

    return results1[1]


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


if __name__ == '__main__':
        for dataset_name in opt.dataset_names:
            opt.dataset_name = dataset_name
            for model_name in opt.model_names:
                opt.model_name = model_name
                if not os.path.exists(opt.save):
                    os.makedirs(opt.save)
                log_file_name = opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + timestamp + '.txt'
                opt.f = open(log_file_name, 'w')
                print(opt.dataset_name + '\t' + opt.model_name)
                train()
                print('\n')
                opt.f.close()
                torch.cuda.empty_cache()