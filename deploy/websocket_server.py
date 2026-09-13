"""Minimal websocket transport for policy inference."""

import asyncio
import http
import logging
import os
import signal
import traceback

import websockets.asyncio.server as _server
import websockets.frames

from deploy.client import msgpack_numpy
from deploy.client.image_codec import decompress_images
from deploy.policy import Policy

logger = logging.getLogger(__name__)


def infer_observation(policy, obs: dict) -> dict:
    """Decode transport-only fields and invoke the policy."""
    if "images" in obs:
        obs["images"] = decompress_images(obs["images"])
    action_prefix = obs.pop("action_prefix", None)
    prefix_length = obs.pop("prefix_length", None)
    noise = obs.pop("noise", None)
    return policy.infer(
        obs,
        noise=noise,
        action_prefix=action_prefix,
        prefix_length=prefix_length,
    )


class WebsocketPolicyServer:
    """Receive observations, run one policy, and return action chunks."""

    def __init__(
        self,
        policy: Policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = metadata or {}

        level = logging.INFO if os.environ.get("DEPLOY_VERBOSE") else logging.WARNING
        logging.getLogger("websockets.server").setLevel(level)

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("Policy server interrupted")

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        shutdown = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown.set)
        try:
            async with _server.serve(
                self._handle,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                ping_interval=300,
                ping_timeout=600,
                close_timeout=120,
                process_request=_health_check,
            ):
                await shutdown.wait()
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)

    async def _handle(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))

        while True:
            try:
                raw = await websocket.recv()
                obs = msgpack_numpy.unpackb(raw)
                result = infer_observation(self.policy, obs)
                packed = packer.pack(result)
                await websocket.send(packed)
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                return
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Inference failed. Traceback sent in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
