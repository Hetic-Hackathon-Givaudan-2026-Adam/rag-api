"""
Voix de synthèse locale avec Piper (open source, fonctionne sans Internet).
Les modèles de voix (.onnx + .onnx.json) sont dans data/voix/ ; pour en ajouter :
    python -m piper.download_voices --download-dir data/voix fr_FR-siwis-medium
"""
import io
import os
import re
import threading
import wave

DOSSIER_VOIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "voix")

# Noms lisibles des voix connues (les autres gardent leur identifiant).
NOMS = {
    "fr_FR-siwis-medium": "Siwis · voix féminine",
    "fr_FR-siwis-low": "Siwis · voix féminine (légère)",
    "fr_FR-tom-medium": "Tom · voix masculine",
    "fr_FR-upmc-medium": "UPMC · voix féminine",
    "fr_FR-gilles-low": "Gilles · voix masculine",
    "fr_FR-mls-medium": "MLS · voix masculine",
}
PREFEREE = "fr_FR-siwis-medium"

_voix_chargees = {}
_verrou = threading.Lock()


def lister_voix():
    """Voix installées dans data/voix (la voix préférée en premier)."""
    if not os.path.isdir(DOSSIER_VOIX):
        return []
    ids = sorted(f[:-5] for f in os.listdir(DOSSIER_VOIX)
                 if f.endswith(".onnx") and os.path.exists(os.path.join(DOSSIER_VOIX, f + ".json")))
    ids.sort(key=lambda i: i != PREFEREE)
    return [{"id": i, "nom": NOMS.get(i, i)} for i in ids]


def disponible():
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return bool(lister_voix())


def _voix(identifiant):
    from piper import PiperVoice
    with _verrou:
        if identifiant not in _voix_chargees:
            _voix_chargees[identifiant] = PiperVoice.load(os.path.join(DOSSIER_VOIX, identifiant + ".onnx"))
        return _voix_chargees[identifiant]


def pour_la_voix(texte):
    """Texte lisible à l'oral : sans identifiants (FR026), symboles ni puces."""
    texte = re.sub(r"\s*[—-]\s*FR\d{3}", "", texte)
    texte = re.sub(r"\(FR\d{3}\)", "", texte)
    texte = texte.replace("™", "").replace("…", "...")
    texte = re.sub(r"^\s*-\s+", "", texte, flags=re.M)
    return re.sub(r"\s+", " ", texte).strip()


def synthetiser(texte, identifiant=None):
    """Renvoie le son (WAV) du texte lu par la voix demandée (ou la voix préférée)."""
    voix_dispo = [v["id"] for v in lister_voix()]
    if not voix_dispo:
        raise RuntimeError("Aucune voix Piper installée dans data/voix.")
    identifiant = identifiant if identifiant in voix_dispo else voix_dispo[0]
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as fichier:
        _voix(identifiant).synthesize_wav(pour_la_voix(texte)[:3000], fichier)
    return tampon.getvalue()
