"""Blocking websocket inference client.

Adapted from Physical Intelligence's openpi client
(https://github.com/Physical-Intelligence/openpi, Apache-2.0); see the
Licenses section of the repo README. Modified to add JPEG compression.
"""

import logging
import time

import websockets.sync.client

from deploy.client import msgpack_numpy
from deploy.client.image_codec import compress_images as _compress_images


class WebsocketClientPolicy:
    """Sends observations to a WebsocketPolicyServer and returns its inference result."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        compress_images: bool = False,
    ) -> None:
        self._compress_images = compress_images
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(
        self,
    ) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    ping_interval=300,
                    ping_timeout=600,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        """Compress, pack, send, receive, and unpack one inference request."""
        if self._compress_images and "images" in obs:
            obs = dict(obs)
            obs["images"] = _compress_images(obs["images"])
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()
