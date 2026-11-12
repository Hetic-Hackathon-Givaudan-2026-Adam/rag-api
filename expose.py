"""Expose l'API locale et son Swagger avec ngrok."""

import argparse
import json
import os
import sys
from threading import Event
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from pyngrok import conf, ngrok
from pyngrok.exception import PyngrokError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Port de l'API (8000 par défaut).")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Le port doit être compris entre 1 et 65535.")

    upstream = f"http://127.0.0.1:{args.port}"
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(f"{upstream}/health", timeout=5) as response:
            health = json.load(response)
        if not isinstance(health, dict) or "statut" not in health:
            raise ValueError("Réponse /health inattendue.")
    except (URLError, TimeoutError, ValueError):
        print(
            f"API indisponible sur {upstream}. Lance d'abord uvicorn main:app --port {args.port}.",
            file=sys.stderr,
        )
        return 1

    if health["statut"] != "ok":
        print("L'API répond. Le moteur RAG indique une erreur : consulte /health avant de tester /ask.")

    config = conf.PyngrokConfig(auth_token=os.getenv("NGROK_AUTHTOKEN"))
    try:
        tunnel = ngrok.connect(addr=upstream, proto="http", bind_tls=True, pyngrok_config=config)
    except (PyngrokError, OSError):
        print(
            "Ouverture ngrok impossible. Vérifie ton compte, ton jeton avec ngrok config add-authtoken "
            "et ta connexion Internet.",
            file=sys.stderr,
        )
        return 1

    try:
        print(f"API publique : {tunnel.public_url}", flush=True)
        print(f"Swagger      : {tunnel.public_url}/docs", flush=True)
        print(f"Test santé   : {tunnel.public_url}/health", flush=True)
        print("Garde ce terminal ouvert. Ctrl+C ferme ce tunnel.", flush=True)
        Event().wait()
    except KeyboardInterrupt:
        print("\nFermeture du tunnel.")
    finally:
        ngrok.disconnect(tunnel.public_url, pyngrok_config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
