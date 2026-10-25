"""Inference for the recorded S0 FCN checkpoint; no pretrained downloads."""
from pathlib import Path
import io
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision.models.segmentation import fcn_resnet50
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


class Segmenter:
    def __init__(self, checkpoint=None):
        self.checkpoint = Path(checkpoint or ROOT / 'checkpoints/S0/best.pt')
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'找不到模型：{self.checkpoint}')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        # Older checkpoints include TorchVersion and NumPy RNG state.
        # Allow only those known types; never fall back to unrestricted pickle.
        from torch.torch_version import TorchVersion
        from numpy.core.multiarray import _reconstruct
        allowed = [TorchVersion, np.dtype, np.ndarray, _reconstruct,
                   type(np.dtype('uint32'))]
        with torch.serialization.safe_globals(allowed):
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        metadata = saved['metadata']
        if metadata['spec']['task'] != 'semantic' or metadata['spec']['model'] != 'fcn':
            raise ValueError('此演示只支持 S0 FCN 语义分割权重。')
        self.height = int(metadata['config']['height'])
        self.width = int(metadata['config']['width'])
        self.model = fcn_resnet50(weights=None, weights_backbone=None,
                                  num_classes=19, aux_loss=False)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.eval()
        self.device = 'cpu'
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
        info = dict(model='FCN-ResNet50', task='semantic_segmentation', device=self.device,
                    input_size=[self.width,self.height], output_size=list(image.size),
                    seconds=round(time.perf_counter()-start,3), opacity=opacity,
                    classes=classes,
                    note='像素占比不是物体数量或准确率。类别 ID 为 Cityscapes trainId 0–18。')
        return dict(original=png_bytes(image), segmentation=png_bytes(color),
                    overlay=png_bytes(overlay), mask=png_bytes(Image.fromarray(mask)), info=info)
