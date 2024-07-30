from itertools import chain
import zlib


def unzip(file_path):
    dec = zlib.decompressobj(zlib.MAX_WBITS + 32)
    with open(file_path, 'rb') as f:
        for chunk in f:
            rv = dec.decompress(chunk)
            if rv:
                f.write(rv)
        if dec.unused_data:
            f.write(dec.flush())


def download_unzip(url: str, file_path: str):
    dec = zlib.decompressobj(0)            
    with urlopen(url) as stream:
         with open(file_path, 'wt') as f:
            for chunk in stream:
                rv = dec.decompress(chunk)
                if rv:
                    f.write(rv)
            if dec.unused_data:
                f.write(dec.flush())


def download_unzip(url: str, file_path: str):
    with urlopen(url) as stream:
         with open(file_path, 'wt') as f:
            for chunk in stream:
                rv = zlib.decompress(chunk, wbits=0)
                if rv:
                    f.write(rv)
            if dec.unused_data:
                f.write(dec.flush())

