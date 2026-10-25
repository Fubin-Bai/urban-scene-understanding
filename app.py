"""Local-only web interface and command line segmentation."""
import argparse
import base64
import io
import json
from pathlib import Path
import threading
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image, UnidentifiedImageError
from segmentation import ROOT, Segmenter, read_image


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, segmenter):
        super().__init__(address, Handler)
        self.segmenter = segmenter
        self.inference_lock = threading.Lock()
        self.download_lock = threading.Lock()
        self.downloads = OrderedDict()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type='application/json; charset=utf-8', filename=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/':
            self.reply(200, (ROOT/'web/index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path.startswith('/download/'):
            parts = self.path.split('/')
            if len(parts) != 4:
                self.reply(404,b'{"error":"Not found"}'); return
            with self.server.download_lock:
                cached = self.server.downloads.get(parts[2])
                artifact = cached.get(parts[3]) if cached else None
            if artifact is None:
                self.reply(404,b'{"error":"Result expired; run prediction again"}'); return
            mime = 'application/json' if parts[3] == 'result.json' else 'image/png'
            self.reply(200,artifact,mime,parts[3])
        elif self.path == '/api/health':
            self.reply(200, json.dumps(dict(ready=True,model='FCN-ResNet50',device='cpu')).encode())
        else:
            self.reply(404, b'{"error":"Not found"}')

    def do_POST(self):
        if self.path != '/api/predict':
            self.reply(404,b'{"error":"Not found"}'); return
        origin = self.headers.get('Origin')
        expected = {f'http://127.0.0.1:{self.server.server_port}',
                    f'http://localhost:{self.server.server_port}'}
        if origin and origin not in expected:
            self.reply(403,b'{"error":"Origin not allowed"}'); return
        if self.headers.get_content_type() != 'application/json':
            self.reply(415,b'{"error":"Expected application/json"}'); return
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 29_000_000:
                raise ValueError('请选择小于 20 MB 的图片。')
            request = json.loads(self.rfile.read(length))
            data = base64.b64decode(request['image'], validate=True)
            opacity = float(request.get('opacity',0.5))
            if not 0 <= opacity <= 1: raise ValueError('透明度必须在 0 和 1 之间。')
            image, original_size = read_image(data)
        except (ValueError, KeyError, TypeError, UnidentifiedImageError, OSError,
                Image.DecompressionBombError) as exc:
            self.reply(400,json.dumps({'error':str(exc) or '无法读取图片。'},ensure_ascii=False).encode());return
        if not self.server.inference_lock.acquire(blocking=False):
            self.reply(429,json.dumps({'error':'模型正在处理其他图片，请稍后重试。'},ensure_ascii=False).encode());return
        try:
            result = self.server.segmenter.predict(image,opacity)
            result['info']['original_size'] = list(original_size)
            payload = {key:'data:image/png;base64,'+base64.b64encode(value).decode()
                       for key,value in result.items() if key != 'info'}
            payload['info'] = result['info']
            token = uuid.uuid4().hex
            artifacts = {key+'.png':value for key,value in result.items() if key != 'info'}
            artifacts['result.json'] = json.dumps(result['info'],ensure_ascii=False,indent=2).encode()
            with self.server.download_lock:
                self.server.downloads[token] = artifacts
                while len(self.server.downloads) > 3:
                    self.server.downloads.popitem(last=False)
            payload['downloads'] = {name:f'/download/{token}/{name}' for name in artifacts}
            self.reply(200,json.dumps(payload,ensure_ascii=False).encode())
        except Exception:
            import traceback
            traceback.print_exc()
            self.reply(500,json.dumps({'error':'预测失败，请查看启动终端中的错误。'},ensure_ascii=False).encode())
        finally:
            self.server.inference_lock.release()


def main():
    parser=argparse.ArgumentParser(description='Street scene semantic segmentation')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/S0/best.pt')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--image',type=Path,help='Run once without the browser')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/prediction')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    print('正在加载 FCN-ResNet50 权重…',flush=True)
    model=Segmenter(args.checkpoint)
    if args.image:
        image,size=read_image(args.image.read_bytes())
        result=model.predict(image)
        result['info']['original_size']=list(size)
        args.output.mkdir(parents=True,exist_ok=True)
        for key,value in result.items():
            if key!='info': (args.output/f'{key}.png').write_bytes(value)
        (args.output/'result.json').write_text(json.dumps(result['info'],ensure_ascii=False,indent=2))
        print('结果已保存：',args.output)
        return
    try: server=DemoServer(('127.0.0.1',args.port),model)
    except OSError as exc:
        raise SystemExit(f'端口 {args.port} 无法使用。尝试 python3 app.py --port {args.port+1}\n{exc}')
    url=f'http://127.0.0.1:{args.port}'
    print(f'模型已就绪：{url}\n按 Ctrl+C 停止。图片只在本机处理。',flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
