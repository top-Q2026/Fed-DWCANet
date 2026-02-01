import numpy as np
import torch
from skimage import measure
from scipy.optimize import linear_sum_assignment
import time
from thop import profile


class mIoU():
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        true_img = np.zeros(predits.shape)
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    true_img[coord_image[m].coords[:, 0], coord_image[m].coords[:, 1]] = 1
                    del coord_image[m]
                    break

        self.dismatch_pixel += (predits - true_img).sum()
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        if self.target == 0:
            Final_PD = 0.0
        else:
            Final_PD = self.PD / self.target
        if self.all_pixel == 0:
            Final_FA = 0.0
        else:
            Final_FA = self.dismatch_pixel / self.all_pixel
        return float(Final_PD), float(Final_FA)

    def reset(self):
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class OPDC_Matcher:
    def __init__(self, dist_threshold=3):
        self.dist_threshold = dist_threshold

    def get_components(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        labeled_mask = measure.label(mask, connectivity=2)
        props = measure.regionprops(labeled_mask)
        return labeled_mask, props

    def match(self, pred_mask, gt_mask):
        pred_label, pred_props = self.get_components(pred_mask)
        gt_label, gt_props = self.get_components(gt_mask)

        M = len(gt_props)
        N = len(pred_props)

        if M == 0 or N == 0:
            return [], set(range(M)), set(range(N)), [], pred_props

        MAX_VAL = 1e6
        dist_matrix = np.full((M, N), MAX_VAL)
        valid_overlap = np.zeros((M, N), dtype=bool)
        iou_matrix = np.zeros((M, N))

        for m in range(M):
            for n in range(N):
                gt_cent = np.array(gt_props[m].centroid)
                pred_cent = np.array(pred_props[n].centroid)
                dist = np.linalg.norm(gt_cent - pred_cent)
                dist_matrix[m, n] = dist

                gt_idx_mask = (gt_label == gt_props[m].label)
                pred_idx_mask = (pred_label == pred_props[n].label)
                intersection = np.logical_and(gt_idx_mask, pred_idx_mask).sum()
                union = np.logical_or(gt_idx_mask, pred_idx_mask).sum()
                iou = intersection / (union + 1e-6)
                iou_matrix[m, n] = iou

                if iou >= 0.5:
                    valid_overlap[m, n] = True

        cost_matrix_1 = dist_matrix.copy()
        cost_matrix_1[~valid_overlap] = MAX_VAL
        row_ind, col_ind = linear_sum_assignment(cost_matrix_1)

        matched_pairs = []
        matched_gt = set()
        matched_pred = set()
        matched_ious = []

        for r, c in zip(row_ind, col_ind):
            if valid_overlap[r, c]:
                matched_pairs.append((r, c))
                matched_gt.add(r)
                matched_pred.add(c)
                matched_ious.append(iou_matrix[r, c])

        unmatched_gt = set(range(M)) - matched_gt
        unmatched_pred = set(range(N)) - matched_pred

        if len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
            u_gt_list = list(unmatched_gt)
            u_pred_list = list(unmatched_pred)
            sub_dist = dist_matrix[np.ix_(u_gt_list, u_pred_list)]
            sub_cost = sub_dist.copy()
            sub_cost[sub_cost >= self.dist_threshold] = MAX_VAL

            r_sub, c_sub = linear_sum_assignment(sub_cost)

            for r, c in zip(r_sub, c_sub):
                if sub_cost[r, c] < MAX_VAL:
                    real_gt_idx = u_gt_list[r]
                    real_pred_idx = u_pred_list[c]
                    matched_pairs.append((real_gt_idx, real_pred_idx))
                    matched_gt.add(real_gt_idx)
                    matched_pred.add(real_pred_idx)
                    matched_ious.append(iou_matrix[real_gt_idx, real_pred_idx])

        final_unmatched_gt = set(range(M)) - matched_gt
        final_unmatched_pred = set(range(N)) - matched_pred

        return matched_pairs, final_unmatched_gt, final_unmatched_pred, matched_ious, pred_props


class IRSTDMetrics:
    def __init__(self, threshold=0.5):
        self.matcher = OPDC_Matcher(dist_threshold=3)
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.iou_list = []
        self.pix_acc_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_pixel = 0
        self.tgt_tp_sum = 0
        self.tgt_fp_sum = 0
        self.tgt_fn_sum = 0
        self.tgt_iou_sum = 0.0
        self.fp_area_sum = 0.0
        self.total_image_area = 0.0

    def update(self, preds, labels):
        if len(preds.shape) == 3: preds = preds.unsqueeze(1)
        if len(labels.shape) == 3: labels = labels.unsqueeze(1)

        preds_binary = (preds > self.threshold).float()
        labels_binary = (labels > self.threshold).float()
        intersection_tensor = (preds_binary * labels_binary).sum(dim=(1, 2, 3))
        pred_area_tensor = preds_binary.sum(dim=(1, 2, 3))
        label_area_tensor = labels_binary.sum(dim=(1, 2, 3))
        union_tensor = pred_area_tensor + label_area_tensor - intersection_tensor
        correct_tensor = (preds_binary == labels_binary).sum(dim=(1, 2, 3))
        total_pixels_tensor = torch.numel(preds_binary[0])

        self.total_inter += intersection_tensor.sum().item()
        self.total_union += union_tensor.sum().item()
        self.total_correct += correct_tensor.sum().item()
        self.total_label += label_area_tensor.sum().item()
        self.total_pixel += (total_pixels_tensor * preds_binary.shape[0])

        batch_size = preds_binary.shape[0]
        for b in range(batch_size):
            inter = intersection_tensor[b].item()
            union = union_tensor[b].item()
            p_area = pred_area_tensor[b].item()
            l_area = label_area_tensor[b].item()
            corr = correct_tensor[b].item()
            self.pix_acc_list.append(corr / total_pixels_tensor)
            iou = inter / (union + 1e-6) if union > 0 else 1.0
            self.iou_list.append(iou)
            prec = inter / (p_area + 1e-6) if p_area > 0 else (0.0 if l_area > 0 else 1.0)
            self.precision_list.append(prec)
            rec = inter / (l_area + 1e-6) if l_area > 0 else 1.0
            self.recall_list.append(rec)
            f1 = 2 * prec * rec / (prec + rec + 1e-6)
            self.f1_list.append(f1)

        np_preds = preds_binary.detach().cpu().squeeze(1).numpy().astype(int)
        np_labels = labels_binary.detach().cpu().squeeze(1).numpy().astype(int)

        for i in range(batch_size):
            p = np_preds[i]
            g = np_labels[i]

            matches, un_gt, un_pred, matched_ious, pred_props = self.matcher.match(p, g)
            tp_t = len(matches)
            fp_t = len(un_pred)
            fn_t = len(un_gt)

            self.tgt_tp_sum += tp_t
            self.tgt_fp_sum += fp_t
            self.tgt_fn_sum += fn_t

            fp_area = sum([pred_props[idx].area for idx in un_pred])
            self.fp_area_sum += fp_area
            self.total_image_area += p.size

    def compute(self):
        global_iou = self.total_inter / (self.total_union + 1e-6)
        mean_iou = np.mean(self.iou_list) if self.iou_list else 0.0
        gt_total = self.tgt_tp_sum + self.tgt_fn_sum
        global_pd = self.tgt_tp_sum / (gt_total + 1e-6)
        global_fa = self.fp_area_sum / (self.total_image_area + 1e-6)

        return {
            "IoU": float(global_iou),
            "nIoU": float(mean_iou),
            "PixAcc": np.mean(self.pix_acc_list) if self.pix_acc_list else 0.0,
            "Precision": np.mean(self.precision_list) if self.precision_list else 0.0,
            "Recall": np.mean(self.recall_list) if self.recall_list else 0.0,
            "Pixel_F1": np.mean(self.f1_list) if self.f1_list else 0.0,
            "Pd": float(global_pd),
            "Fa": float(global_fa * 1e6),
        }


class EfficiencyMetrics:
    def __init__(self, model, input_size=(1, 1, 256, 256), device='cuda'):
        self.model = model
        self.input_size = input_size
        self.device = device

    def get_complexity(self):
        if profile is None: return 0.0, 0.0
        dummy_input = torch.randn(self.input_size).to(self.device)
        self.model.eval()
        try:
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            return params / 1e6, flops / 1e9
        except Exception as e:
            print(f"Complexity Error: {e}")
            return 0.0, 0.0

    def get_speed(self, loops=100):
        dummy_input = torch.randn(self.input_size).to(self.device)
        self.model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for _ in range(20): _ = self.model(dummy_input)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(loops): _ = self.model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()

        mem_use = torch.cuda.max_memory_allocated() / 1024 / 1024
        total_time = end - start
        fps = loops / total_time
        time_ms = (total_time / loops) * 1000

        return time_ms, fps, mem_use