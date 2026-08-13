"""Websocket inference client + wire codecs for the deploy policy server.

Modules:
- ``websocket_client_policy``: blocking request/response inference client.
- ``async_websocket_client``: non-blocking client with real-time action
  chunking (RTC) support for overlapped execution and inference.
- ``msgpack_numpy``: msgpack (de)serialization with NumPy array support.
- ``image_codec``: JPEG compression of camera images for remote inference.

``msgpack_numpy`` and the skeleton of ``websocket_client_policy`` are adapted
from Physical Intelligence's openpi client (Apache-2.0); see the Licenses
section of the repo README.
"""
