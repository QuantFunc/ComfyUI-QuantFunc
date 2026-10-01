"""Terms the shipped plugin must never carry, shared by its death rules (user 2026-09-19 rule, reaffirmed 2026-09-25: the plugin
layer shows no implementation detail — nobody reading the plugin's files may learn how the engine gets faster). Every term is
stored as the SHA-256 of its lowercase form, so this file and the tests that use it never spell one out.

  words    — single words (split on anything but letters / digits)
  phrases  — two adjacent words, joined by one space
  cjk      — CJK sequences, matched as substrings of a CJK run
  keys     — option keys the plugin never sends (exact strings)
  names    — retired helper / setter / query names (compared lowercase)
  mode_ids — identifiers of the withdrawn choice (compared whole, underscores kept)
  desc_*   — the engine's inner vocabulary, kept out of every user-visible text (see desc_hits)
"""
import hashlib
import re

WORDS = {"6c83126bfcb0c11f963495f9102d4a1ea0a90a7e1a65c67878bc7e1e18beac67",
         "dba75bc4aff64c430e14530d50edd0409cf150911adbc56a74fd2e4f7dde11fa",
         "0fedead8d392392e76e93c4dba310b588e17ec9dadc44bd1fe90707480c8905c",
         "ce18d968699ef0d99874a7c331b710129e8268e34ff4db49c7d13cc9d79aa4ac",
         "d5682e53de65da1058fe5cb35d57cc6c8ac4c4bf4a2d8942c3e7594899dea1d1",
         "3e7572d78adbc32b21c34d8d78bb2e86e1880c9b4dc4d498aeee95b022ae8e09",
         "a664b1ba599eca15be95bbc51779d2ca2f6d99ef817406ccb8bff394e1052012",
         "69fda41cdd1be7c1af65bc03bd7a3f26ccfac19563a43f235f50c646a9b6a039",
         "6a14c3ef2763b29b1d98a553f1b21c4243079103a93b4869be6d199940526a0f",
         "c048264deeca0c17ac1dc2787bb1bfd76e2fea8067d2d2e648cadbbf282dc023",
         "b9b792f2220b142bba53ca6d9a341fa52c232b95d14503b2b41f8f7555684cbe",
         "d2085368c7618790df0aec2a4d68579ef602e0340c9bbb04c26f254a8b8392ca",
         "69b7c7c5771013e93babfbe522b6b696c6d1fc9422cc324f4c2461ec14b5e22d",
         "3ba3535ad3bf448182c53fc1110afd8d8b636261849bbb856117c2066e0dd7cf",
         "61e07efb5a559720c2ed6d685b7b1978d57565c9cfe05cf53ee4de74566ab449",
         "fa9c9d57522cfd42bdf4aacde0bf6eaa29975e0082df0df63f0737a57c7013ad",
         "1766cfd31cc1e261e84ac3c0819db3cf9c1689899f9014d09071477a800e56cf",
         "2ff5409a7626b6d7c92d76922436bd22fec7f31f0fd7a70d275b72b179b3d431",
         "4b6e15c59dd3b31f697f7737626d292bc3e840fde731fe5041dbb8473a0a6259"}
PHRASES = {"1688a85cced8e8a1baf724e4c7146a82549da3f55976d9b35ce1e1ebfc438565",
           "a7ad900b7e20fed9e367fb4050904e77b6d09aa2533ff68613adc5187bd4d910",
           "990aee7a23ba60e86fd336d7eb315d622b7ec3a9641be0095eb2adae201b450a",
           "335b2c08a5a263a4ff3610913ba0379b9d5d6a42a5813f0cb3132709040a6a0a",
           "5b2e56073ae4adc12a1b69c1947408ac964728315eed28d0cd1ae3c5077192f8",
           "8a0607681df6c3034008ee26d9946e6a0efa0cbcf1ab106b7dd1aa76318eaba1",
           "b12fcee9cf3b9f86f9bb77ea49828c0a7c18bcbffbe195e7667fd0902423c55f",
           "8c2f9f22bfa05974ec8ecdf4d042352e8b1cd36c63168792315ba24a093a85fd",
           "7160fffc0e7c1bd41c8e9b777ea2c7be0930bbbace9ceab5c37ba35f42898f52",
           "2af7909ca08f18facc556624b02e1a5c683bb0f557137b1ef7e0028fc457715c",
           "b04d1a614603cacdff3c6ee2463d646a2b3d37ce55fbfcc9309d0ae0d80e7116",
           "fab5d10d64647a8d97bc74a1047f99c15cf3f531c9333e1c3e6099cfbba31a2e",
           "26e1d998f3c3d40adf911c67b32769bafda60392250ba3d46661459d989e4173",
           "8c9b00e5521c444e53c51b1c1b7b24be68134f511fcd95bc22088e53869f887b",
           "13be34d01fd1a97fad66ba1b5b57a78450657fd5047d5e6992d02ebb4916dd73",
           "fc5deac2955c637b1ec9c70f62a69989c10bf480d3aedfca59d56b6a90129457",
           "18603030e95b9931e43f9767c6a5b0b9a90bfa11ad2f573b1f2f90e30aae315c",
           "e6f0cbe284b1f904829505949b398a43a6c007762f056c57c009ee9ae5997ac9",
           "289dbf61db3d9a4cc8ffa71c2cbe21c7ddeffe849b9863615d57af8ff6856851",
           "9a1f0751459e010a1660b598235d0a85206c22ce07db70257e1a8a44f022b21e"}
CJK = {"38c0085ccac6f88d3d8149a3a1c728223ce611dd8fc0a64a2b408f57b5e50af8",
       "3e190d67411970ccd784f5234145f38dd697463259fb197db0a6e0aa3a1273e1",
       "a59f86b6dc3f100915c4ded2e30f1c8c791b22d91d7f1780ec61d945464e9d18",
       "632f8920323e639d10d75cd4426acc9dd46618c3e5b8894a9a7d1c637f11378f"}
KEYS = {"dd4c884c284e4fc13b840bdd0009a33c3b827f8d253888be4a733c41ed015f59",
        "e738a900f4f0ac9d99d1d5499785baa300956b272eb988e9d689aa37092c5ca2",
        "98d8221d46f242fc53e2f34c1e4dcb4d2b2adfb1b232cfbcb71a2349960716a2",
        "ee59cfbbc6306cb50b04dee3f8556ae9792286f2f7e5be7d2932e398a68b874a"}
NAMES = {"ce9962986c9514161c7502c835e494a28ef8df63ad0a024a82cb5ddb52fa11c1",
         "f928cf9a099d81382d322cf6a744101bc4d24f7683d7622fb5623e0ac274d8bb",
         "2441f2a330b08884549924039149c2616df59b3d88200be09d6fdfb035169ff8",
         "decb5524e4f5a6efd0d3938c186e26eae937a5a7792613825703ca16c01f71dc",
         "af99c9a48f339deba0f430cf736860c416f2ef7f4b7f3e9cc061d457dbc528fd",
         "1acbd27318b72979bd24ba3ce6a9ee80befd8e2fcb614994c88d614c2c31af37",
         "1b36767bc8ff3a19bfeebc3c1e1862eac9ca9d2a7b551524f033f6bfd703e5c6",
         "c57e40533dd31ec26172183b32570ea12cc16166d9a8fb199a0258c3180abbeb",
         "23a2bf07c653a9ff7d6f8568a02298362c451c7ae943b27bf025a2c6900c3b85",
         "d3e74a2dd1153543664d89c5b5eabcbc8b5a396e1c36ffca13fddaa8b32b4fa3"}
MODE_IDS = {"c4507b5deaaa5a459f1c17077fbd54066d0d0a3a01244ce7e051a610431b4b30",
            "762399977373b8078726eee3cc6c5ea2b7f4d05eb58e8c249fb9969b0c5e45fe"}
# the engine's inner vocabulary a USER-VISIBLE text (a loader DESCRIPTION or tooltip, a workflow note, the README's switch
# paragraph) must not carry (user 「介绍上不要透露技术细节」): whole words (a widget name such as step_cache stays one word),
# parts of a word (so plural and derived forms count) and two adjacent words (any separator)
DESC_WORDS = {"00e987128dfe081a06deaf11dc1a4edd6020b43a0e2f51c5b0bb5bf3ed027574",
              "3e64cc41cf8e07b43936d7bc4eaf8bdb8e2abcec1f44c7b915ac0a0c27ebc41a",
              "7e5d4325a44714fc86a9b6989e41e966a7297bfe7e913c15fb9ab19588e33d61",
              "851e43bd44a2c3d30e5f3acadc9240c12d9f1c610dba34761e8c47ba82d1daea",
              "b7595e2a863957fc6c225ed540d335dd1e859dd3032451c2e11bceb0329defa6"}
DESC_PARTS = {"1766cfd31cc1e261e84ac3c0819db3cf9c1689899f9014d09071477a800e56cf",
              "19f12f3f0c0be9255f4a6a8864d6f2c96263edbe4b0809ee2917ce3c47f61d85",
              "3c469e9d6c5875d37a43f353d4f88e61fcf812c66eee3457465a40b0da4153e0",
              "57bc8116d6476c2e608c560f260b6545716a61ed1841abf8eb1719e7889cb45a",
              "5fd8731f4b31ea4e894d65cd98d5cc27c5583a7aae69ed215ca1e17c877f9957",
              "68c2f9ee314749c05c96df0cad305b0972506d78bb9b23c942cf805b274236c6",
              "6923dd1bc0460082c5d55a831908c24a282860b7f1cd6c2b79cf1bc8857c639c",
              "69b7c7c5771013e93babfbe522b6b696c6d1fc9422cc324f4c2461ec14b5e22d",
              "6c8b4535ccc87f19061c4646189e33d78f01c8b63dc4e3cb2f630b1796ee93b6",
              "80da2f0c42b25856bb005f7d43347be096b1de0678986109b02338da172d9051",
              "9ebd2f36b6588db268eedf6b44184ba9593b05a91f617717ac4e3345ec43e6af",
              "cb1525bced78da2c03c42fe15bf15663b584566ef6244ff91d892caa011fec1e",
              "cb52babcb5840ff564069421e29bd6322850f6c42f345adcff112ac1a83a93fb"}
DESC_PHRASES = {"18603030e95b9931e43f9767c6a5b0b9a90bfa11ad2f573b1f2f90e30aae315c",
                "2af7909ca08f18facc556624b02e1a5c683bb0f557137b1ef7e0028fc457715c",
                "878c2beeef869106a3765875103792ca105df7b4f580e742711e102c8f793f40"}
# the two words the switch's own texts (the QI-2.1 workflow notes, the README's switch paragraph) must not carry either;
# those texts do talk about sampler steps and model files, so they are checked against this set, not the DESC_* sets
NOTE_PARTS = {"3c469e9d6c5875d37a43f353d4f88e61fcf812c66eee3457465a40b0da4153e0",
              "68c2f9ee314749c05c96df0cad305b0972506d78bb9b23c942cf805b274236c6"}
# a model FILE name may carry any token (a published checkpoint's name is not the plugin's wording)
_MODEL_FILE = re.compile(r"[\w.+-]+\.(?:safetensors|gguf|ckpt|pth?|bin|onnx)\b", re.I)
_CJK_RUN = re.compile(r"[㐀-鿿]+")


def h(s):
    return hashlib.sha256(s.lower().encode("utf-8")).hexdigest()


def term_hits(text, words=WORDS, phrases=PHRASES, cjk=CJK):
    """The banned terms in `text`, as they are written there (empty = clean)."""
    text = _MODEL_FILE.sub(" ", text)
    toks = re.findall(r"[a-z0-9]+", text.lower())
    hits = [t for t in toks if h(t) in words]
    hits += [f"{a} {b}" for a, b in zip(toks, toks[1:]) if h(f"{a} {b}") in phrases]
    for run in _CJK_RUN.findall(text):
        hits += [run[i:i + n] for n in (2, 3, 4) for i in range(len(run) - n + 1) if h(run[i:i + n]) in cjk]
    return hits


def mode_id_hits(text):
    """Identifiers of the withdrawn choice written in `text` (underscores kept, so ordinary English words never match)."""
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if h(t) in MODE_IDS]


def desc_hits(text, words=DESC_WORDS, parts=DESC_PARTS, phrases=DESC_PHRASES):
    """The inner-vocabulary terms in a user-visible `text`, as written there (empty = clean)."""
    low = text.lower()
    toks = re.findall(r"[a-z0-9_]+", low)   # word characters, as a regex \b sees them
    hits = [t for t in toks if h(t) in words]
    hits += [t for t in toks if any(h(t[i:j]) in parts for i in range(len(t)) for j in range(i + 3, len(t) + 1))]
    sub = re.findall(r"[a-z0-9]+", low)
    return hits + [f"{a} {b}" for a, b in zip(sub, sub[1:]) if h(f"{a} {b}") in phrases]


def selftest():
    """The matcher, both ways, on canaries that are no real term: each kind is found, and clean text stays clean."""
    cw, cp, cc = {h("zqvcanary")}, {h("zqv canary")}, {h("鸮鹲鸸")}
    assert term_hits("a Zqvcanary here", cw, set(), set()) == ["zqvcanary"]
    assert term_hits("the zqv-canary pair", set(), cp, set()) == ["zqv canary"]
    assert term_hits("文字鸮鹲鸸文字", set(), set(), cc) == ["鸮鹲鸸"]
    assert term_hits("zqvcanary-v2.safetensors", cw, set(), set()) == []      # a model file name is not wording
    assert term_hits("nothing to see", cw, cp, cc) == [] and mode_id_hits("the best quality") == []
    dw, dp = {h("zqvword")}, {h("zqvpart")}
    assert desc_hits("a Zqvword, not zqvwords nor zqvword_cache", dw, set(), set()) == ["zqvword"]
    assert desc_hits("un-ZQVPARTed and zqvparts", set(), dp, set()) == ["zqvparted", "zqvparts"]
    assert desc_hits("the zqv_canary and zqv canary", set(), set(), cp) == ["zqv canary", "zqv canary"]
    assert desc_hits("nothing to see", dw, dp, cp) == []
    return True
