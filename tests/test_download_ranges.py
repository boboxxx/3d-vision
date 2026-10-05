"""Exercise real HTTP ranges, nonaligned resume and checksum rejection."""
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download_kitti',ROOT/'scripts/download_kitti.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class RangeTests(unittest.TestCase):
    def test_parallel_prefix_integrity(self):
        content = bytes(range(256))*83
        etag = hashlib.md5(content).hexdigest()
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass
            def do_HEAD(self):
                self.send_response(200)
                self.send_header('Content-Length',str(len(content)))
                self.send_header('ETag',etag)
                self.end_headers()
            def do_GET(self):
                begin,end = map(int,self.headers['Range'][6:].split('-'))
                requests.append((begin,end))
                self.send_response(206)
                self.send_header('Content-Range',f'bytes {begin}-{end}/{len(content)}')
                self.send_header('Content-Length',str(end-begin+1))
                self.send_header('ETag',etag)
                self.end_headers()
                self.wfile.write(content[begin:end+1])
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            url = 'http://127.0.0.1:'+str(server.server_port)+'/archive.zip'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'archive.zip'
                path.with_suffix('.zip.part').write_bytes(content[:777])
                # Corrupted cached segment must be redownloaded, not trusted by size.
                segments = path.with_suffix('.zip.segments')
                segments.mkdir()
                segment = segments/'777-4872.bin'
                segment.write_bytes(b'x'*4096)
                segment.with_suffix('.json').write_text(json.dumps({'sha256':'incorrect'}))
                identity = downloader.download(url,path,segments=4,segment_bytes=4096)
                self.assertEqual(path.read_bytes(),content)
                self.assertEqual(identity['sha256'],hashlib.sha256(content).hexdigest())
                self.assertEqual(sorted(requests),downloader.remaining_ranges(777,len(content),4096))
                # Never mix segmented downloads from an archive with a different ETag.
                path.unlink()
                (segments/'identity.json').write_text(json.dumps({'url':url,'size':len(content),'etag':'changed'}))
                with self.assertRaisesRegex(RuntimeError,'identity changed'):
                    downloader.download(url,path,segments=4,segment_bytes=4096)
        finally:
            server.shutdown()
            server.server_close()


if __name__=='__main__':
    unittest.main()
