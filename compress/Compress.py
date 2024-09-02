from compress import GM
import torch.nn.utils.prune as prune
import time
import torch.nn as nn
import torch
from ultralytics.nn.modules import *
import yaml
from ultralytics import YOLO


class PruneHandler():
    def __init__(self, model, compression_ratio, method, cfg_output_path, prune_type='ALL'):
        self.model = model
        self.ckpt = model.ckpt['model']
        self.model.cpu()
        self.cr = compression_ratio
        self.method = method
        self.cfg_output_path = cfg_output_path
        self.model.to('cpu')  # cuda cannot convert to numpy
        self.remain_index_out = {}
        self.prune_type = prune_type

    def prune(self):
        if self.method == 'GM':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not '22' in name:
                        if isinstance(module, nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not any(
                            name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '22']):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if any(name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L1':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not '22' in name:
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not any(
                            name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '22']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if any(name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L2':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not '22' in name:
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not any(
                            name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '22']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if any(name.startswith(n + '.') for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=1-self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

    def reconstruct(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.training = False

        detect_in_channels = []
        concat = {}
        remain_out_channels = [0, 1, 2]
        for name, module in self.model.model.model.named_modules():
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '16', '19']:
                    num = int(name)
                    offset = self.model.model.model[num].conv.weight.shape[0]
                    remain_out_channels = module.recon(remain_out_channels)
                if name in ['16', '19']:
                    concat[name] = [remain_out_channels, offset]
            elif isinstance(module, C2f):
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels)
                if name in ['15', '18', '21']:
                    detect_in_channels.append(remain_out_channels)
                elif name in ['4', '6', '12']:
                    concat[name] = [remain_out_channels, offset]

            elif isinstance(module, SPPF):
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels)
                concat[name] = [remain_out_channels, offset]

            elif isinstance(module, Detect):
                remain_out_channels = module.recon(detect_in_channels)

            elif isinstance(module, Concat):
                if name == '11':
                    concat['6'][0] = [x + concat['9'][1] for x in concat['6'][0]]
                    remain_out_channels = concat['9'][0] + concat['6'][0]
                elif name == '14':
                    concat['4'][0] = [x + concat['12'][1] for x in concat['4'][0]]
                    remain_out_channels = concat['12'][0] + concat['4'][0]
                elif name == '17':
                    concat['12'][0] = [x + concat['16'][1] for x in concat['12'][0]]
                    remain_out_channels = concat['16'][0] + concat['12'][0]
                elif name == '20':
                    concat['9'][0] = [x + concat['19'][1] for x in concat['9'][0]]
                    remain_out_channels = concat['19'][0] + concat['9'][0]

    def model_to_yaml(self):
        from_ = -1
        repeats = 1
        yaml_dict = {}
        yaml_dict["nc"] = 8
        yaml_dict["scales"] = {'prune': [1, 1, 1024]}
        yaml_dict["backbone"] = []
        yaml_dict["head"] = []

        for name, module in self.model.ckpt['model'].model.named_modules():
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '16', '19']:
                    args = [module.conv.out_channels, module.conv.kernel_size[0], module.conv.stride[0]]
                    layer = [from_, repeats, type(module).__name__, args]
                    if name in ["16", "19"]:
                        yaml_dict["head"].append(layer)
                    else:
                        yaml_dict["backbone"].append(layer)

            elif isinstance(module, C2f):
                args = [module.cv2.conv.out_channels, module.m[0].add, module.m[0].cv1.conv.in_channels,
                        module.m[0].cv1.conv.out_channels, module.cv1.conv.out_channels]
                layer = [from_, len(module.m), type(module).__name__, args]
                if name in ["12", "15", "18", "21"]:
                    yaml_dict["head"].append(layer)
                else:
                    yaml_dict["backbone"].append(layer)

            elif isinstance(module, SPPF):
                args = [module.cv2.conv.out_channels, module.m.kernel_size, module.cv1.conv.out_channels]
                layer = [from_, repeats, type(module).__name__, args]
                yaml_dict["backbone"].append(layer)

            elif isinstance(module, Detect):
                args = [yaml_dict["nc"], module.cv3[0][0].conv.in_channels, module.cv3[0][0].conv.out_channels]
                layer = [[15, 18, 21], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, Concat):
                args = [1]
                if name == '11':
                    # import pdb; pdb.set_trace()
                    layer = [[from_, 6], repeats, type(module).__name__, args]
                elif name == '14':
                    layer = [[from_, 4], repeats, type(module).__name__, args]
                elif name == '17':
                    layer = [[from_, 12], repeats, type(module).__name__, args]
                elif name == '20':
                    layer = [[from_, 9], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, nn.Upsample):
                args = ['None', module.scale_factor, module.mode]
                layer = [from_, repeats, "nn." + type(module).__name__, args]
                yaml_dict["head"].append(layer)
        yaml_str = yaml.dump(yaml_dict)
        with open(f'/home/yjlee/yolo_pruning/{self.cfg_output_path}/c_bestmodel.yaml', "w") as file:
            file.write(yaml_str)

    def compress_yolov8(self):
        print('Pruning...')
        start = time.time()
        self.prune()
        self.reconstruct()
        torch.save(self.model, f'./{self.cfg_output_path}/best_model_prune.pt')
        self.model_to_yaml()
        print('Done')
        # import pdb; pdb.set_trace()
        print(f'time : {time.time() - start}')
        return YOLO(f'./{self.cfg_output_path}/c_bestmodel.yaml').load(f'./{self.cfg_output_path}/best_model_prune.pt')