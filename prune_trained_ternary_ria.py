#!/usr/bin/env python3
"""Post-training RIA N:M pruning of an already-trained CIFAR-10 ternary Spikformer.

Uses model_ternary.py or model_ternary_atten.py and the checkpoint's args.yaml.
No optimizer and no fine-tuning.
"""
import argparse
import csv
import json
import math
import importlib
import inspect
from pathlib import Path
from itertools import islice

import torch
import yaml
from torch.utils.data import DataLoader
from torch.nn.utils import prune
from torchvision import datasets
from timm.data import create_transform



def _ria_layers(model):
    layers = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'SpikingSelfAttention':
            for branch in ('q_conv', 'k_conv', 'v_conv'):
                layers[f'{name}.{branch}'] = getattr(module, branch)
        elif module.__class__.__name__ == 'MLP':
            for branch in ('mlp1_conv', 'mlp2_conv'):
                layers[f'{name}.{branch}'] = getattr(module, branch)
    return layers


def _reset_spiking_state(model):
    for module in model.modules():
        reset = getattr(module, 'reset', None)
        if callable(reset):
            reset()


def ria_scores(weight, input_l2, activation_power=0.5):
    """RIA Eq. (3), using PyTorch's [output, input] weight convention."""
    if not math.isfinite(activation_power) or activation_power <= 0:
        raise ValueError('RIA activation power must be finite and positive')
    magnitude = weight.detach().float().flatten(1).abs()
    norms = input_l2.to(device=weight.device, dtype=torch.float32)
    if norms.shape != (magnitude.shape[1],):
        raise ValueError('Activation norms must match the input channels')
    if not torch.isfinite(magnitude).all() or not torch.isfinite(norms).all() or (norms < 0).any():
        raise ValueError('Weights and activation norms must be finite; norms must be nonnegative')
    tiny = torch.finfo(magnitude.dtype).tiny
    relative = (magnitude / magnitude.sum(0, keepdim=True).clamp_min(tiny)
                + magnitude / magnitude.sum(1, keepdim=True).clamp_min(tiny))
    return relative * norms.pow(activation_power).unsqueeze(0)


def nm_mask(scores, n, m):
    if (n, m) not in ((4, 8), (8, 16)):
        raise ValueError('Supported N:M patterns are 4:8 and 8:16')
    if scores.ndim != 2 or scores.shape[1] % m:
        raise ValueError('Input channel count must be divisible by M')
    if not torch.isfinite(scores).all():
        raise ValueError('RIA scores must be finite')
    groups = scores.reshape(scores.shape[0], -1, m)
    selected = groups.argsort(dim=-1, descending=True, stable=True)[..., :n]
    return torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, selected, True).reshape_as(scores)


@torch.no_grad()
def prune_ria_nm(model, calibration_loader, n=4, m=8, num_batches=32, activation_power=0.5):
    if (n, m) not in ((4, 8), (8, 16)):
        raise ValueError('Supported N:M patterns are 4:8 and 8:16')
    layers = _ria_layers(model)
    if not layers or num_batches < 1:
        raise ValueError('RIA requires target layers and at least one calibration batch')
    for name, layer in layers.items():
        if layer.groups != 1 or any(size != 1 for size in layer.kernel_size) or layer.in_channels % m:
            raise ValueError(f'{name}: requires a pointwise convolution with input channels divisible by {m}')
        if hasattr(layer, 'weight_mask'):
            raise ValueError(f'{name}: already pruned; start from dense weights')
    sums = {}
    handles = []
    modes = {module: module.training for module in model.modules()}

    def collect(name):
        def hook(layer, inputs):
            x = inputs[0].detach().float()
            squares = x.square().sum(tuple(dim for dim in range(x.ndim) if dim != 1))
            if name in sums:
                sums[name].add_(squares)
            else:
                sums[name] = squares
        return hook

    try:
        model.eval()
        for name, layer in layers.items():
            handles.append(layer.register_forward_pre_hook(collect(name)))
        parameter = next(model.parameters())
        for images, _ in islice(calibration_loader, num_batches):
            _reset_spiking_state(model)
            try:
                model(images.to(device=parameter.device, dtype=parameter.dtype))
            finally:
                _reset_spiking_state(model)
    finally:
        for handle in handles:
            handle.remove()
        _reset_spiking_state(model)
        for module, training in modes.items():
            module.training = training
    masks = {}
    for name, layer in layers.items():
        if name not in sums:
            raise ValueError(f'{name}: no calibration samples collected')
        scores = ria_scores(layer.weight, sums[name].sqrt(), activation_power)
        masks[name] = nm_mask(scores, n, m).reshape_as(layer.weight)
    for name, layer in layers.items():
        prune.custom_from_mask(layer, 'weight', masks[name])
        layer.weight_orig.mul_(layer.weight_mask)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=None, help='Default: args.yaml beside the checkpoint')
    p.add_argument('--checkpoint', required=True, help='Existing TRAINED, DENSE ternary checkpoint')
    p.add_argument('--model-source', choices=('auto', 'model_ternary', 'model_ternary_atten'), default='auto',
                   help='auto selects model_ternary_atten when training config includes attn_scale_mode')
    p.add_argument('--ria-nm', choices=['4:8', '8:16'], required=True)
    p.add_argument('--output-dir', default='results_ria_posttrain')
    p.add_argument('--calibration-batches', type=int, default=32)
    p.add_argument('--calibration-batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2 if torch.cuda.is_available() else 0)
    p.add_argument('--limit-eval-batches', type=int, default=0,
                   help='Limit each evaluation for smoke tests; 0 evaluates the complete test set')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def create_spikformer(cfg, model_source='auto'):
    if model_source == 'auto':
        model_source = 'model_ternary_atten' if 'attn_scale_mode' in cfg else 'model_ternary'
    architecture = importlib.import_module(model_source)
    kwargs = dict(
        drop_rate=0., drop_path_rate=0.2,
        img_size_h=int(cfg['img_size']), img_size_w=int(cfg['img_size']),
        patch_size=int(cfg['patch_size']), embed_dims=int(cfg['dim']),
        num_heads=int(cfg['num_heads']), mlp_ratios=float(cfg['mlp_ratio']),
        in_channels=3, num_classes=int(cfg['num_classes']), qkv_bias=False,
        depths=int(cfg['depths']), sr_ratios=1, T=int(cfg['time_step']),
        ternary_threshold=float(cfg.get('ternary_threshold', 1.)),
        ternary_surrogate_alpha=float(cfg.get('ternary_surrogate_alpha', 2.)),
        attn_scale=float(cfg.get('attn_scale', .125)),
        ternary_v_mode=cfg.get('ternary_v_mode', 'ternary'),
        score_norm=cfg.get('score_norm', 'none'),
        score_scale=float(cfg.get('score_scale', 1.)),
    )
    signature = inspect.signature(architecture.vit_snn)
    if 'attn_scale_mode' in signature.parameters:
        kwargs['attn_scale_mode'] = cfg.get('attn_scale_mode', 'global')
    if cfg.get('qk_norm', 'none') != 'none' and 'qk_norm' not in signature.parameters:
        raise ValueError('Training uses QK normalization but current model source does not support it')
    if 'qk_norm' in signature.parameters:
        kwargs['qk_norm'] = cfg.get('qk_norm', 'none')
    model = architecture.vit_snn(**kwargs)
    model._ria_model_source = str(Path(architecture.__file__).resolve())
    return model


def attention_scale_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name.endswith(('.log_attn_scale', '.fixed_attn_scale'))}


def load_weights(model, path, require_dense=True):
    # Only load checkpoint files from trusted sources.
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError('Expected a checkpoint dictionary or state_dict')
    state = obj.get('state_dict', obj.get('model', obj))
    if not isinstance(state, dict):
        raise TypeError('Checkpoint does not contain a state_dict')
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    layers = _ria_layers(model)
    if not layers:
        raise RuntimeError('No QKV/MLP RIA layers found in selected model')
    masks = {}
    for name, layer in layers.items():
        mask = state.get(f'{name}.weight_mask')
        if mask is None:
            if f'{name}.weight_orig' in state:
                raise RuntimeError(f'{name}: pruning checkpoint is missing its weight_mask')
            continue
        if mask.shape != layer.weight.shape or not ((mask == 0) | (mask == 1)).all():
            raise RuntimeError(f'{name}: invalid weight_mask')
        masks[name] = mask
    sparse = any(not mask.all().item() for mask in masks.values())
    if sparse and require_dense:
        raise RuntimeError('Checkpoint already contains RIA masks. Use a DENSE trained checkpoint.')
    if sparse:
        # Install the standard PyTorch pruning parametrization before loading.
        for name in masks:
            layer = layers[name]
            prune.custom_from_mask(layer, 'weight', torch.ones_like(layer.weight))
            # Also accept the older model's weight + weight_mask checkpoint layout.
            if f'{name}.weight_orig' not in state:
                state[f'{name}.weight_orig'] = state.pop(f'{name}.weight')
    else:
        # All-one masks describe dense weights and need no parametrization.
        for name in masks:
            state.pop(f'{name}.weight_mask')
            if f'{name}.weight_orig' in state:
                state[f'{name}.weight'] = state.pop(f'{name}.weight_orig')
    model.load_state_dict(state, strict=True)
    print(f'Loaded {"sparse" if sparse else "dense"} checkpoint: {path} ({len(layers)} RIA layers)')


def dataset_loader(cfg, train, batch_size, workers):
    size = int(cfg.get('img_size', 32))
    mean = tuple(cfg.get('mean', (0.4914, 0.4822, 0.4465)))
    std = tuple(cfg.get('std', (0.247, 0.2435, 0.2616)))
    transform = create_transform(
        input_size=(3, size, size), is_training=False,
        interpolation=cfg.get('interpolation', 'bicubic'),
        mean=mean, std=std, crop_pct=float(cfg.get('crop_pct', 1.0)))
    ds = datasets.CIFAR10(root=str(cfg['data_dir']), train=train,
                          transform=transform, download=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=False)


@torch.no_grad()
def evaluate(model, loader, device, limit_batches=0):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    try:
        batches = islice(loader, limit_batches) if limit_batches else loader
        for x, y in batches:
            _reset_spiking_state(model)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise ValueError('Evaluation produced non-finite logits')
            loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    finally:
        _reset_spiking_state(model)
    if not total:
        raise ValueError('No evaluation samples')
    return {'top1': 100. * correct / total, 'loss': loss_sum / total, 'samples': total}


def inspect_nm(model, n, m):
    rows = []
    total_kept = total_weights = 0
    for name, layer in _ria_layers(model).items():
        mask = layer.weight_mask.reshape(layer.weight.shape[0], -1, m)
        if not torch.all(mask.sum(-1) == n):
            raise AssertionError(f'Bad {n}:{m} pattern at {name}')
        kept = int(mask.sum())
        count = layer.weight.numel()
        total_kept += kept
        total_weights += count
        rows.append({'layer': name, 'total': count, 'kept': kept,
                     'mask_sparsity': (count - kept) / count})
    return rows, total_kept, total_weights


def main():
    args = parse_args()
    if min(args.calibration_batches, args.calibration_batch_size, args.eval_batch_size) < 1:
        raise ValueError('Calibration batch count and all batch sizes must be positive')
    if args.workers < 0 or args.limit_eval_batches < 0:
        raise ValueError('Workers and evaluation batch limit must be nonnegative')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser() if args.config else checkpoint.parent / 'args.yaml'
    with config.open('r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Config must contain a YAML mapping')
    if cfg.get('ternary_v_mode', 'ternary') != 'ternary':
        raise ValueError('This script expects ternary Q/K/V')
    torch.manual_seed(int(cfg.get('seed', 42)))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    print(f'Device: {device}; config: {config}; threshold: {cfg.get("ternary_threshold", 1.0)}')
    if args.limit_eval_batches:
        print(f'SMOKE TEST: metrics use at most {args.limit_eval_batches} test batches, not the full test set.')
    model = create_spikformer(cfg, args.model_source).to(device)
    print('Model source:', model._ria_model_source, flush=True)
    load_weights(model, checkpoint)
    scales_before = attention_scale_state(model)
    calibration = dataset_loader(cfg, train=True,
                                 batch_size=args.calibration_batch_size, workers=args.workers)
    test_loader = dataset_loader(cfg, train=False,
                                 batch_size=args.eval_batch_size, workers=args.workers)
    print('Evaluating trained dense ternary model ...')
    before = evaluate(model, test_loader, device, args.limit_eval_batches)
    print(f"Dense     | Top-1={before['top1']:.2f}% | Loss={before['loss']:.4f}")
    n, m = map(int, args.ria_nm.split(':'))
    print(f'Pruning once with RIA {n}:{m}, using {args.calibration_batches} training batches ...')
    prune_ria_nm(model, calibration, n=n, m=m,
                 num_batches=args.calibration_batches,
                 activation_power=float(cfg.get('ria_activation_power', 0.5)))
    layer_rows, kept, total = inspect_nm(model, n, m)
    scales_after = attention_scale_state(model)
    if scales_before.keys() != scales_after.keys() or any(
            not torch.equal(value, scales_after[name]) for name, value in scales_before.items()):
        raise AssertionError('Attention scale parameters unexpectedly changed during pruning')
    print('Attention scales preserved:', len(scales_before), flush=True)
    print('Evaluating pruned ternary model WITHOUT fine-tuning ...')
    after = evaluate(model, test_loader, device, args.limit_eval_batches)
    print(f"RIA {n}:{m}  | Top-1={after['top1']:.2f}% | Loss={after['loss']:.4f}")
    print(f"Top-1 change: {after['top1'] - before['top1']:+.2f} percentage points")
    print(f'RIA-layer sparsity: {(1 - kept/total)*100:.2f}% ({total-kept}/{total})')
    all_parameters = sum(p.numel() for p in model.parameters())
    overall_masked_fraction = (total-kept) / all_parameters
    print(f'Overall parameter mask fraction: {overall_masked_fraction*100:.2f}%')
    out = Path(args.output_dir) / args.ria_nm.replace(':', '_')
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / 'ternary_ria_pruned.pth'
    torch.save({'state_dict': model.state_dict(), 'config': cfg,
                'ria_nm': args.ria_nm, 'source_checkpoint': str(checkpoint),
                'model_source': model._ria_model_source}, save_path)
    report = {'ria_nm': args.ria_nm, 'source_checkpoint': str(checkpoint),
              'model_source': model._ria_model_source, 'attention_scales_preserved': True,
              'attention_scales': {name: (value.exp() if name.endswith('.log_attn_scale') else value).tolist()
                                   for name, value in scales_after.items()},
              'config': str(config), 'device': str(device),
              'calibration_batches': args.calibration_batches,
              'calibration_batch_size': args.calibration_batch_size,
              'limit_eval_batches': args.limit_eval_batches,
              'dense': before, 'pruned': after,
              'top1_change': after['top1'] - before['top1'],
              'target_mask_sparsity': 1 - kept / total,
              'overall_parameter_mask_fraction': overall_masked_fraction}
    with (out / 'metrics.json').open('w', encoding='utf8') as f:
        json.dump(report, f, indent=2)
    with (out / 'layer_sparsity.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'total', 'kept', 'mask_sparsity'])
        writer.writeheader()
        writer.writerows(layer_rows)
    print(f'Saved pruned weights: {save_path}')


if __name__ == '__main__':
    main()
