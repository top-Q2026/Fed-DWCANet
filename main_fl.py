import argparse
import os
import time
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader

# --- Import your project modules ---
from model.DWCANet import DWCANet
from dataset import TrainSetLoader, TestSetLoader
from loss import SoftIoULoss
from metrics import mIoU
from utils import seed_pytorch


def get_args():
    parser = argparse.ArgumentParser(description="Federated Learning for IRSTD (Weighted FedAvg)")
    # Basic Configuration
    parser.add_argument("--model_name", default='DWCANet', type=str)
    parser.add_argument("--dataset_dir", default='./datasets', type=str)
    parser.add_argument("--save_dir", default='./logs/FL_Exp_Weighted', type=str, help="Root directory for saving weights")
    parser.add_argument("--num_rounds", type=int, default=150, help="Total communication rounds for FL")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local training epochs for clients")
    parser.add_argument("--clients", default=['NUAA-SIRST', 'NUDT-SIRST'], nargs='+', help="List of clients")
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--patchSize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resume", default='./log/FL_Exp_Weighted/Server/Server_Round_100.pth.tar', type=str)
    parser.add_argument("--start_round", default=0, type=int, help="Start counting from which round")
    return parser.parse_args()


args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_fl_checkpoint(state, save_path):
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(state, save_path)


def fed_avg(weights_and_counts):
    total_samples = sum([count for _, count in weights_and_counts])
    base_weights, base_count = weights_and_counts[0]
    w_avg = copy.deepcopy(base_weights)
    first_coefficient = base_count / total_samples
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * first_coefficient
    for i in range(1, len(weights_and_counts)):
        w_local, n_local = weights_and_counts[i]
        coefficient = n_local / total_samples

        for k in w_avg.keys():
            w_avg[k] += w_local[k] * coefficient

    print(f"[FedAvg] Aggregation completed. Total samples: {total_samples}")
    return w_avg

class Client:
    def __init__(self, client_name):
        self.name = client_name
        self.save_dir = os.path.join(args.save_dir, f'Client_{client_name}')

        # 1. Load training dataset
        print(f"[Client Init] {client_name} is loading training data...")
        self.train_dataset = TrainSetLoader(args.dataset_dir, client_name, args.patchSize)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=2,
                                       drop_last=True)

        self.num_samples = len(self.train_dataset)
        print(f"-> Number of samples: {self.num_samples}")

        self.test_dataset = TestSetLoader(args.dataset_dir, client_name, client_name)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        self.best_mIoU = 0.0

    def local_train(self, global_weights, round_idx):
        net = DWCANet().to(device)
        net.load_state_dict(global_weights)
        net.train()

        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
        criterion = SoftIoULoss()

        for epoch in range(args.local_epochs):
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                labels = (labels > 0.5).float()

                optimizer.zero_grad()
                preds = net(imgs)

                if isinstance(preds, list):
                    loss = sum([criterion(p, labels) for p in preds]) / len(preds)
                else:
                    loss = criterion(preds, labels)

                loss.backward()
                optimizer.step()

        current_mIoU = self.evaluate(net)
        print(f"> Client [{self.name}] Round {round_idx} finished. mIoU: {current_mIoU:.4f}")

        client_weights = net.state_dict()

        save_path = os.path.join(self.save_dir, f'{self.name}_Round_{round_idx}.pth.tar')
        save_fl_checkpoint({'round': round_idx, 'state_dict': client_weights, 'mIoU': current_mIoU}, save_path)

        if current_mIoU > self.best_mIoU:
            self.best_mIoU = current_mIoU
            best_save_path = os.path.join(self.save_dir, f'{self.name}_best.pth.tar')
            save_fl_checkpoint({'round': round_idx, 'state_dict': client_weights, 'best_mIoU': self.best_mIoU},
                               best_save_path)
            print(f"Client [{self.name}] New Record! Best mIoU updated to: {self.best_mIoU:.4f}")

        return client_weights, self.num_samples

    def evaluate(self, net):
        net.eval()
        evaluator = mIoU()
        with torch.no_grad():
            for img, gt, size, _ in self.test_loader:
                img = img.to(device)
                pred = net(img)
                if isinstance(pred, list): pred = pred[-1]
                pred = (pred > args.threshold).cpu()
                gt = gt[:, :, :size[0], :size[1]]
                pred = pred[:, :, :size[0], :size[1]]
                evaluator.update(pred, gt)
        _, score = evaluator.get()
        net.train()
        return score


def init_system():
    seed_pytorch(args.seed)
    global_model = DWCANet().to(device)

    if args.resume and os.path.exists(args.resume):
        print(f"!!! Resuming from checkpoint: {args.resume} !!!")
        checkpoint = torch.load(args.resume)
        global_model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Checkpoint not found. Training from scratch.")

    clients = [Client(name) for name in args.clients]
    print(f"\n=== System Initialization Complete ===")
    return global_model, clients


def run_round(round_idx, global_model, clients):
    print(f"=== Round {round_idx}/{args.num_rounds} Start ===")
    global_weights = global_model.state_dict()

    local_results = []

    for client in clients:
        w, n_samples = client.local_train(global_weights, round_idx)
        local_results.append((w, n_samples))

    print("  [Server] Aggregating weights (Weighted FedAvg)...")
    new_global_weights = fed_avg(local_results)
    global_model.load_state_dict(new_global_weights)

    # 3. Save global model
    save_path = os.path.join(args.save_dir, 'Server', f'Server_Round_{round_idx}.pth.tar')
    save_fl_checkpoint({'round': round_idx, 'state_dict': new_global_weights}, save_path)
    print(f"  [Server] Global model saved.\n")


def main():
    global_model, clients = init_system()
    # Note: If resuming, you might want to adjust the range start based on args.start_round
    for round_idx in range(1, args.num_rounds + 1):
        run_round(round_idx, global_model, clients)
    print("=== Federated Learning Finished ===")



if __name__ == '__main__':
    main()