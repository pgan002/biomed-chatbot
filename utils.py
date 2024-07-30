import os
import zipfile
import zlib
from urllib.request import urlretrieve, urlopen


def download_unzip(url: str, file_path: str):
    print(f'Downloading from {url}...')
    zip_file_path = os.path.join(os.path.dirname(file_path), os.path.basename(url))
    urlretrieve(url, zip_file_path)
    with zipfile.ZipFile(zip_file_path) as f:
        f.extractall(out_file_path)
    os.remove(zip_file_path)


def download_unzip_streaming(url: str, file_path: str):
    '''Broken: dec.decompress() throws error regardless of wbits param'''
    print(f'Downloading from {url}...')
    dec = zlib.decompressobj(zlib.MAX_WBITS + 32)
    with urlopen(url) as stream:
         with open(file_path, 'wt') as f:
            for chunk in stream:
                rv = dec.decompress(chunk)
                if rv:
                    f.write(rv)
            if dec.unused_data:
                f.write(dec.flush())

