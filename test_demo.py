"""Integration checks against a running local demo: python3 test_demo.py."""
import base64
import io
import json
import unittest
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
from PIL import Image

BASE='http://127.0.0.1:8765'
ROOT=Path(__file__).resolve().parent

class DemoIntegration(unittest.TestCase):
    def post(self,payload):
        request=urllib.request.Request(BASE+'/api/predict',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        return urllib.request.urlopen(request,timeout=60)

    def test_health(self):
        with urllib.request.urlopen(BASE+'/api/health') as response:
            self.assertTrue(json.load(response)['ready'])

    def test_real_checkpoint_prediction(self):
        # A recorded figure is a functional smoke input, not an accuracy benchmark.
        data=(ROOT/'assets/semantic-S0.png').read_bytes()
        with self.post({'image':base64.b64encode(data).decode(),'opacity':0}) as response:
            result=json.load(response)
        images={key:Image.open(io.BytesIO(base64.b64decode(result[key].split(',')[1])))
                for key in ['original','segmentation','overlay','mask']}
        self.assertEqual(len({image.size for image in images.values()}),1)
        self.assertTrue(np.array_equal(np.asarray(images['original']),np.asarray(images['overlay'])))
        mask=np.asarray(images['mask'])
        self.assertEqual(mask.ndim,2)
        self.assertLessEqual(int(mask.max()),18)
        self.assertGreater(len(np.unique(mask)),1)
        from segmentation import PALETTE
        self.assertTrue(np.array_equal(PALETTE[mask],np.asarray(images['segmentation'])))
        with urllib.request.urlopen(BASE+result['downloads']['mask.png']) as download:
            self.assertIn('attachment',download.headers['Content-Disposition'])
            self.assertEqual(download.read(),base64.b64decode(result['mask'].split(',')[1]))
        with urllib.request.urlopen(BASE+result['downloads']['result.json']) as download:
            self.assertEqual(json.load(download)['classes'],result['info']['classes'])
        classes=result['info']['classes']
        self.assertAlmostEqual(sum(c['percent'] for c in classes),100,places=5)
        for c in classes:self.assertEqual(c['pixels'],int((mask==c['id']).sum()))

    def test_invalid_image(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({'image':base64.b64encode(b'not an image').decode()})
        self.assertEqual(caught.exception.code,400)

    def test_invalid_opacity(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({'image':'','opacity':2})
        self.assertEqual(caught.exception.code,400)

if __name__=='__main__':unittest.main()
