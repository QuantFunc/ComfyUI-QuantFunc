"""The API key a QuantFunc loader takes from its own field (user 2026-09-27: a valid key typed there wins; config.json is
then not read).

ComfyUI saves every widget value into the workflow (saved and exported files, copy/paste, the browser's autosave, the
`workflow` text of every image and video) and every prompt input into the `prompt` text of every image and video, its
history and "Export (API)". It has no secret widget a custom node can use: its only secret channel is hardcoded to the
two comfy.org keys (execution.SENSITIVE_EXTRA_DATA_KEYS). So the key is never a widget value or a prompt value:

  * web/quantfunc_api_key.js: the field is a password input that is never saved (serialize = false). When a prompt is
    queued it POSTs the key to /quantfunc/api_key and queues only the reference it gets back;
  * this module keeps the key in this ComfyUI process, behind that reference, and turns the reference back into the key
    for the loader (field_key). A reference means nothing to another process, or after a restart.
"""
import hashlib
import hmac
import os
import re

# The QuantFunc service mints every key as "qf_" + 64 lowercase hex digits (32 random bytes: quantfunc-server
# models/api_key.go, generateKey, its only mint path). A field of any other shape is a typo: refused, never sent.
KEY_FORMAT = re.compile(r"qf_[0-9a-f]{64}")
REF_PREFIX = "qfk:"
_REF_HEX = 32          # 128 bits of an HMAC-SHA256: unguessable
_MAX_FIELD = 1024      # longer than any key: the route refuses it, which also bounds what one request can store
_SECRET = os.urandom(32)   # per process, so a reference found in an image names nothing another process holds
_KEYS = {}                 # reference -> the text the field POSTed
# ponytail: one entry per distinct text POSTed while this process runs, never evicted. Only someone who can queue
# prompts on this ComfyUI can grow it; cap it if that ever matters.


def remember(key):
    """Keep `key` in this process; return the reference a prompt may carry instead of it. The same key gives the same
    reference for the whole process, so an unchanged field leaves the loader's inputs, and ComfyUI's cache, unchanged."""
    ref = REF_PREFIX + hmac.new(_SECRET, key.encode("utf-8"), hashlib.sha256).hexdigest()[:_REF_HEX]
    _KEYS[ref] = key
    return ref


def field_key(value):
    """The loader field's key, from the value its prompt carries: None when the field is empty (or absent: without the
    field's script there is no field), which leaves the key to QUANTFUNC_API_KEY / config.json as before. Anything else
    that is not a reference to a well-formed key is refused loud; no message ever contains the value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise RuntimeError("qf_native: this loader's api_key input is not text. Leave it to the loader's API key field.")
    key = _KEYS.get(value.strip())
    if key is None and value.strip().startswith(REF_PREFIX):
        raise RuntimeError("qf_native: this prompt's API key was entered in another ComfyUI session (the key is kept "
                           "only in memory, never in the prompt). Enter the key in the loader's API key field again and "
                           "queue the prompt again.")
    if key is None:
        raise RuntimeError("qf_native: this prompt carries an API key as plain text. A key in a prompt is saved into "
                           "every image and video the prompt makes, so it is not used. Enter the key in the loader's "
                           "API key field (it is kept out of saved workflows and images), or put it in config.json.")
    if not KEY_FORMAT.fullmatch(key):
        raise RuntimeError("qf_native: the text in this loader's API key field is not a QuantFunc API key (a key is "
                           "qf_ followed by 64 characters 0-9 and a-f). Paste the whole key from your QuantFunc account, "
                           "or empty the field to use the key in config.json.")
    return key


async def _post_key(request):
    """POST {"key": text} -> {"ref": reference} ("" for an empty key). Never answers with, or logs, the text."""
    from aiohttp import web
    try:
        body = await request.json()
    except ValueError:
        body = None
    key = body.get("key") if isinstance(body, dict) else None
    if not isinstance(key, str) or len(key) > _MAX_FIELD:
        return web.json_response({"error": 'expected a JSON object {"key": "<the API key>"}'}, status=400)
    key = key.strip()
    return web.json_response({"ref": remember(key) if key else ""})


def register_route(routes):
    """Add POST /quantfunc/api_key to ComfyUI's route table (PromptServer.instance.routes)."""
    routes.post("/quantfunc/api_key")(_post_key)
