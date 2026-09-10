# Google FDE source

The three files under `sketching/point_cloud/` are unmodified copies from
`google/graph-mining`, revision `dca52ed1c35ca530d11ea7f76b0bf74634cdeb5c`.
See `upstream.json` for original source hashes and `LICENSE` for Apache 2.0.

`bridge.cc` and `CMakeLists.txt` are RAS integration files. They do not replace
upstream FDE arithmetic. CMake builds a small shared library callable from
Python 3.13 via ctypes, with no dependency on the Python C API, PyTorch, or CUDA.

The first build fetches pinned Abseil, Eigen and Protobuf sources; later builds
reuse them. No Abseil/protobuf compatibility shims or changes to Google code
are used.
