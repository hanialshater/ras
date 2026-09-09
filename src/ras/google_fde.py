"""ctypes adapter to unmodified Google graph-mining FDE C++ functions."""
import ctypes as ct
from pathlib import Path

import numpy as np

REVISION = 'dca52ed1c35ca530d11ea7f76b0bf74634cdeb5c'
CONFIGS = {
    'r20_b5_p8': dict(repetitions=20, bits=5, projection=8, final_dimension=0),
    'r20_b5_p16': dict(repetitions=20, bits=5, projection=16, final_dimension=0),
    'r8_b4_4096': dict(repetitions=8, bits=4, projection=0, final_dimension=4096),
    'raw_r8_b4': dict(repetitions=8, bits=4, projection=0, final_dimension=0),
}


class GoogleFDE:
    def __init__(self, library, dim, *, repetitions=20, bits=5, projection=8,
                 final_dimension=0, seed=7, threads=2):
        if (dim < 1 or repetitions < 1 or not 0 <= bits <= 12 or
                projection < 0 or final_dimension < 0 or not 1 <= threads <= 64 or
                not 0 <= seed < 2**31-repetitions):
            raise ValueError('Invalid official FDE configuration')
        self.dim, self.threads = dim, threads
        self.config = dict(repetitions=repetitions, bits=bits, projection=projection,
                           final_dimension=final_dimension, seed=seed)
        raw_dimension = repetitions*(1 << bits)*(projection or dim)
        self.output_dim = final_dimension or raw_dimension
        if max(raw_dimension, self.output_dim) > 1_000_000:
            raise ValueError('FDE dimensions exceed this diagnostic wrapper limit')
        self.library = ct.CDLL(str(Path(library).resolve()))
        self.function = self.library.fde_encode
        self.function.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ct.c_int64,
            *([ct.c_int]*8), np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ct.POINTER(ct.c_char), ct.c_int]
        self.function.restype = ct.c_int

    def encode_batch(self, values, offsets, *, query):
        values = np.ascontiguousarray(values, dtype=np.float32)
        offsets = np.asarray(offsets)
        if (values.ndim != 2 or values.shape[1] != self.dim or
                offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) < 2 or
                offsets[0] != 0 or offsets[-1] != len(values) or
                np.any(np.diff(offsets.astype(np.int64)) <= 0) or not np.isfinite(values).all()):
            raise ValueError('Expected finite nonempty ragged token matrices')
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        output = np.empty((len(offsets)-1, self.output_dim), dtype=np.float32)
        error = ct.create_string_buffer(2048)
        c = self.config
        status = self.function(values, offsets, len(output), self.dim, c['repetitions'],
            c['bits'], c['projection'], c['final_dimension'], c['seed'], int(query),
            self.threads, output, error, len(error))
        if status or not np.isfinite(output).all():
            raise RuntimeError('Google FDE failed: ' + error.value.decode(errors='replace'))
        return output

    def encode(self, tokens, *, query):
        return self.encode_batch(tokens, np.array([0, len(tokens)]), query=query)[0]
