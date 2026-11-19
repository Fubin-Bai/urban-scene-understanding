"""Checkpoint-format and model-selection tests; no downloaded weights needed."""
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from segmentation import Segmenter, ModelRegistry

class TinySemantic(torch.nn.Module):
    def load_state_dict(self, state, strict=True):
        assert strict
    def forward(self, x):
        return {'out': torch.zeros((x.shape[0],19,*x.shape[-2:]))}

class CheckpointFormats(unittest.TestCase):
    def load(self, metadata, expected=None):
        saved={'metadata':metadata,'model':{}}
        with patch('pathlib.Path.is_file',return_value=True), \
             patch('torch.load',return_value=saved) as loader, \
             patch('segmentation.fcn_resnet50',return_value=TinySemantic()), \
             patch('segmentation.deeplabv3_resnet50',return_value=TinySemantic()):
            model=Segmenter('/tmp/format-test.pt',expected)
        self.assertTrue(loader.call_args.kwargs['weights_only'])
        return model

    def test_legacy_fcn(self):
        m=self.load({'spec':{'task':'semantic','model':'fcn'},
                     'config':{'height':256,'width':512,'epochs':5,'seed':9444}})
        self.assertEqual((m.height,m.width),(256,512))
        self.assertEqual(m.details['epochs'],5)

    def test_v2_deeplab_and_export(self):
        m=self.load({'spec':{'task':'semantic','model':'deeplab','height':512,
                            'width':1024,'epochs':12},'stage':'final','epoch':12,'seed':9444},'deeplab')
        result=m.predict(Image.new('RGB',(32,16)))
        self.assertEqual(result['info']['model'],'DeepLabV3-ResNet50')
        self.assertEqual(result['info']['input_size'],[1024,512])
        self.assertEqual(result['info']['checkpoint']['stage'],'final')
        self.assertEqual(result['info']['output_size'],[32,16])

    def test_reject_wrong_model(self):
        with self.assertRaises(ValueError):
            self.load({'spec':{'task':'semantic','model':'fcn'},
                       'config':{'height':256,'width':512}},'deeplab')

    def test_reject_missing_dimensions(self):
        with self.assertRaises(ValueError):
            self.load({'spec':{'task':'semantic','model':'deeplab'}})

    def test_no_checkpoint_is_not_ready(self):
        with patch('pathlib.Path.is_file',return_value=False):
            registry=ModelRegistry()
            self.assertFalse(registry.describe()['ready'])
            with self.assertRaises(FileNotFoundError):
                registry.predict(Image.new('RGB',(8,8)),model_key='deeplab')

if __name__=='__main__':unittest.main()
