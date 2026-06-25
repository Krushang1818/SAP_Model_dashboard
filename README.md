# VirtuCEO SAP Model Dashboard

Standalone PC2 model server for inference, retraining, model activation, and secure PC1 pairing.

## First-time setup

1. Run `git lfs pull` to download the trained adapters and checkpoints.
2. Create a Python virtual environment and install `requirements.txt`.
3. Copy `.env.example` to `.env` and configure the local base model path.
4. Optionally run `install_cloudflared.ps1` for internet pairing links.
5. Optionally run `enable_model_server_firewall.ps1` as Administrator for LAN access.

## Start PC2

```powershell
.\start_model_server.ps1
```

The launcher opens `http://127.0.0.1:8001/`. Use the connection panel to copy a LAN link, start a temporary Quick Tunnel, or start a configured permanent Cloudflare Tunnel.

The node ID and bearer token are generated on first start in ignored `config/node_identity.json`. Rotating the token invalidates every previously copied pairing link.

## Permanent Cloudflare Tunnel

Create a remotely managed tunnel and public hostname in Cloudflare, routing it to `http://localhost:8001`. Set these values in `.env`:

```dotenv
CLOUDFLARE_TUNNEL_TOKEN=...
CLOUDFLARE_PUBLIC_URL=https://model.example.com
```

The public tunnel exposes only token-protected model APIs. The dashboard and tunnel/token management endpoints accept local PC2 requests only.
