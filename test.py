import argparse
from torch.utils.data import DataLoader
from dataset import *
from metrics import *
import os
import time
import cv2
from model.baseline import *
from model.DWCANet import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")

parser.add_argument("--model_names", default=['DWCANet'], nargs='+',
                    help="model_name: 'baseline','DNANet','DNANetCA','DNANetWT','DWCANet'")
parser.add_argument("--pth_dirs", default=None, nargs='+',
                    help="checkpoint dir, default=None or ['epoch/UIUNet_best.pth.tar','epoch/UIUNet_best.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['IRSTD-1K'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./logs/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
    opt.img_norm_cfg['std'] = opt.img_norm_cfg_std


def print_beautified_table(model_name, dataset_name, acc_results, eff_results):
    print("\n" + "=" * 60)
    print(f" EVALUATION REPORT")
    print(f" Model:   {model_name}")
    print(f" Dataset: {dataset_name}")
    print("=" * 60)
    print(f" {'🏆 Accuracy Metrics':<30} | {'Value':<15}")
    print("-" * 60)
    print(f" {'IoU':<30} | {acc_results['IoU']:<15.6f}")
    print(f" {'nIoU':<30} | {acc_results['nIoU']:<15.6f}")
    print(f" {'F1-Score':<30} | {acc_results['Pixel_F1']:<15.6f}")
    print("-" * 60)
    print(f" {'Pd (Prob. Detection)':<30} | {acc_results['Pd']:<15.6f}")
    print(f" {'Fa (False Alarm)':<30} | {acc_results['Fa']:<15.6f}")
    print("=" * 60)
    print(f" {'⚡ Efficiency Metrics':<30} | {'Value':<15}")
    print("-" * 60)
    print(f" {'Params (M)':<30} | {eff_results['Params (M)']:<15.4f}")
    print(f" {'FLOPs (G)':<30} | {eff_results['FLOPs (G)']:<15.4f}")
    print(f" {'FPS':<30} | {eff_results['FPS']:<15.2f}")
    print(f" {'Latency (ms)':<30} | {eff_results['Latency (ms)']:<15.4f}")
    print(f" {'GPU Memory (MB)':<30} | {eff_results['Peak GPU Memory (MB)']:<15.2f}")
    print("=" * 60 + "\n")


def test():
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    if opt.model_name == 'baseline':
        net = Baseline().cuda()
    elif opt.model_name == 'DWCANet':
        net = DWCANet().cuda()

    try:
        net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device,weights_only= False)['state_dict'])
    net.eval()
    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    eval_metrics = IRSTDMetrics(threshold=opt.threshold)
    print(f"Start testing {opt.model_name} on {opt.test_dataset_name}...")
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = img.cuda()
            gt_mask = gt_mask.cuda()
            pred = net.forward(img)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :].cpu(), size)
            eval_metrics.update(pred, gt_mask)

    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    acc_results = eval_metrics.compute()
    print("Calculating efficiency...")
    eff_tool = EfficiencyMetrics(net, input_size=(1, 1, 256, 256), device='cuda')
    params, flops = eff_tool.get_complexity()
    time_ms, fps, mem = eff_tool.get_speed(loops=50)
    eff_results = {
        'Params (M)': params,
        'FLOPs (G)': flops,
        'FPS': fps,
        'Latency (ms)': time_ms,
        'Peak GPU Memory (MB)': mem
    }
    print_beautified_table(opt.model_name, opt.test_dataset_name, acc_results, eff_results)

    opt.f.write(f"dataset: {opt.test_dataset_name} | model: {opt.model_name}\n")
    opt.f.write(f"IoU: {acc_results['IoU']:.6f} | nIoU: {acc_results['nIoU']:.6f} | \n")
    opt.f.write(f"Pixacc: {acc_results['PixAcc']:.6f} | Precision: {acc_results['Precision']:.6f} | Recall: {acc_results['Recall']:.6f}|\n"
                f" Pd: {acc_results['Pd']:.6f} | Fa: {acc_results['Fa']:.6f}  Pixel_F1: {acc_results['Pixel_F1']:.6f}\n")
    opt.f.write(
        f"Params(M): {eff_results['Params (M)']:.4f} | FLOPs(G): {eff_results['FLOPs (G)']:.4f} | FPS: {eff_results['FPS']:.2f} \n"
        f"| Peak GPU Memory(MB): {eff_results['Peak GPU Memory (MB)']:.2f} | Latency(ms): {eff_results['Latency (ms)']:.2f}\n")
    opt.f.write("-" * 20 + "\n")


if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_best.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_best.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = dataset_name
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')

                        if os.path.exists(opt.pth_dir):
                            pass
                        else:
                            print(f"⚠️ Checkpoint not found: {opt.pth_dir}")
        opt.f.close()

