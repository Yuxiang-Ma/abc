"""Serve the released ABC-DiT policy over websocket."""

import logging
import os

import tyro

from deploy.policy import Policy
from deploy.serve_policy_config import Args, default_args
from deploy.websocket_server import WebsocketPolicyServer


def create_policy(args: Args) -> Policy:
    if not args.policy.checkpoint_path:
        raise ValueError("A checkpoint is required")
    return Policy(args.policy)


def main(args: Args) -> None:
    level = logging.INFO if os.environ.get("DEPLOY_VERBOSE") else logging.WARNING
    logging.basicConfig(level=level, force=True)
    policy = create_policy(args)
    print(
        f"[serve_policy] ABC-DiT steps={args.policy.diffusion_steps} "
        f"fast={args.policy.fast_inference} chunk_len={policy.chunk_len}"
    )
    print(f"[serve_policy] serving on 0.0.0.0:{args.port}")
    WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args, default=default_args()))
