"""Inference for the recorded S0 FCN checkpoint; no pretrained downloads."""
from pathlib import Path
import io
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
CLASSES = [
    ('road', '道路', (128,64,128)), ('sidewalk','人行道',(244,35,232)),
    ('building','建筑',(70,70,70)), ('wall','墙体',(102,102,156)),
    ('fence','围栏',(190,153,153)), ('pole','杆柱',(153,153,153)),
    ('traffic light','交通灯',(250,170,30)), ('traffic sign','交通标志',(220,220,0)),
    ('vegetation','植被',(107,142,35)), ('terrain','地面',(152,251,152)),
    ('sky','天空',(70,130,180)), ('person','行人',(220,20,60)),
    ('rider','骑行者',(255,0,0)), ('car','汽车',(0,0,142)),
    ('truck','卡车',(0,0,70)), ('bus','公交车',(0,60,100)),
    ('train','轨道车辆',(0,80,100)), ('motorcycle','摩托车',(0,0,230)),
    ('bicycle','自行车',(119,11,32)),
]
PALETTE = np.array([x[2] for x in CLASSES], dtype=np.uint8)
Image.MAX_IMAGE_PIXELS = 24_000_000


def png_bytes(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def read_image(data):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError('图片文件不能超过 20 MB。')
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 24_000_000:
            raise ValueError('图片不能超过 2400 万像素。')
        original_size = source.size
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    return image, original_size


MODEL_OPTIONS = {
    'fcn': {'name': 'FCN-ResNet50', 'checkpoint': ROOT / 'checkpoints/S0/best.pt'},
    'deeplab': {'name': 'DeepLabV3-ResNet50', 'checkpoint': ROOT / 'checkpoints/DeepLabV3/model.pt'},
}


class Segmenter:
    def __init__(self, checkpoint=None, expected_model=None):
        self.checkpoint = Path(checkpoint or MODEL_OPTIONS['fcn']['checkpoint'])
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'找不到模型：{self.checkpoint}')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        # Keep loading restricted, including the known legacy RNG metadata types.
        from torch.torch_version import TorchVersion
        from numpy.core.multiarray import _reconstruct
        allowed = [TorchVersion, np.dtype, np.ndarray, _reconstruct,
                   type(np.dtype('uint32'))]
        with torch.serialization.safe_globals(allowed):
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        metadata = saved['metadata']
        spec = metadata['spec']
        model_key = spec.get('model')
        if spec.get('task') != 'semantic' or model_key not in MODEL_OPTIONS:
            raise ValueError('只支持 FCN / DeepLabV3 的 19 类语义分割权重。')
        if expected_model is not None and model_key != expected_model:
            raise ValueError(f'权重模型是 {model_key}，与所选的 {expected_model} 不一致。')
        # Legacy checkpoints keep dimensions in config; v2 keeps them in spec.
        config = metadata.get('config', {})
        self.height = int(spec.get('height', config.get('height', 0)))
        self.width = int(spec.get('width', config.get('width', 0)))
        if not (0 < self.height <= 2048 and 0 < self.width <= 4096):
            raise ValueError('权重缺少有效的输入尺寸。')
        constructor = fcn_resnet50 if model_key == 'fcn' else deeplabv3_resnet50
        self.model = constructor(weights=None, weights_backbone=None,
                                 num_classes=19, aux_loss=False)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.eval()
        self.model_key = model_key
        self.model_name = MODEL_OPTIONS[model_key]['name']
        self.device = 'cpu'
        self.details = {
            'id': model_key, 'name': self.model_name, 'available': True,
            'input_size': [self.width, self.height],
            'epochs': spec.get('epochs', config.get('epochs')),
            'checkpoint_epoch': metadata.get('epoch', saved.get('epoch')),
            'stage': metadata.get('stage', 'legacy'),
            'seed': metadata.get('seed', config.get('seed')),
        }
        del saved

    @torch.inference_mode()
    def predict(self, image, opacity=0.5):
        if not 0 <= opacity <= 1:
            raise ValueError('透明度必须在 0 和 1 之间。')
        start = time.perf_counter()
        resized = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        tensor = TF.normalize(TF.to_tensor(resized),
                              [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        logits = self.model(tensor.unsqueeze(0))['out']
        prediction = F.interpolate(logits, size=(image.height, image.width),
                                   mode='bilinear', align_corners=False).argmax(1)[0]
        mask = prediction.numpy().astype(np.uint8)
        color = Image.fromarray(PALETTE[mask])
        overlay = Image.blend(image, color, opacity)
        counts = np.bincount(mask.ravel(), minlength=19)
        classes = [dict(id=i, name=name, label=label, color=list(rgb),
                        pixels=int(counts[i]), percent=float(counts[i] / mask.size * 100))
                   for i, (name, label, rgb) in enumerate(CLASSES) if counts[i] > 0]
        classes.sort(key=lambda x: x['pixels'], reverse=True)
        info = dict(model=self.model_name, model_id=self.model_key, checkpoint=self.details, task='semantic_segmentation', device=self.device,
                    input_size=[self.width,self.height], output_size=list(image.size),
                    seconds=round(time.perf_counter()-start,3), opacity=opacity,
                    classes=classes,
                    note='像素占比不是物体数量或准确率。类别 ID 为 Cityscapes trainId 0–18。')
        return dict(original=png_bytes(image), segmentation=png_bytes(color),
                    overlay=png_bytes(overlay), mask=png_bytes(Image.fromarray(mask)), info=info)


class ModelRegistry:
    """Keep only the selected network in memory; never download weights implicitly."""
    def __init__(self, checkpoint=None, default_model=None):
        self.paths = {key: value['checkpoint'] for key, value in MODEL_OPTIONS.items()}
        self.active = None
        if checkpoint is not None:
            self.active = Segmenter(checkpoint, default_model)
            self.paths[self.active.model_key] = Path(checkpoint)
            self.default = self.active.model_key
        else:
            self.default = default_model or ('deeplab' if self.paths['deeplab'].is_file() else 'fcn')
            if self.paths[self.default].is_file():
                self.active = Segmenter(self.paths[self.default], self.default)

    def describe(self):
        rows = []
        for key, option in MODEL_OPTIONS.items():
            row = {'id': key, 'name': option['name'], 'available': self.paths[key].is_file()}
            if self.active is not None and self.active.model_key == key:
                row.update(self.active.details)
            if not row['available']:
                row['reason'] = f'尚未提供权重：{self.paths[key].relative_to(ROOT) if self.paths[key].is_relative_to(ROOT) else self.paths[key]}'
            rows.append(row)
        return {'ready': any(row['available'] for row in rows), 'device': 'cpu',
                'model': self.active.model_name if self.active else None,
                'default_model': self.default, 'models': rows}

    def predict(self, image, opacity=0.5, model_key=None):
        key = model_key or self.default
        if key not in self.paths:
            raise ValueError('未知模型，请选择 FCN 或 DeepLabV3。')
        if not self.paths[key].is_file():
            raise FileNotFoundError(f'{MODEL_OPTIONS[key]["name"]} 尚未提供权重。')
        if self.active is None or self.active.model_key != key:
            # Release the old model before loading the other one on CPU.
            self.active = None
            self.active = Segmenter(self.paths[key], key)
        return self.active.predict(image, opacity)
