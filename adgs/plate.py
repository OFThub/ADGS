"""M2 ANPR alt modulu - TR plaka okuma ve DOGRULAMA.

Bu modulun asil isi OCR degil, OCR ciktisini reddetmektir. Yanlis okunan bir
plaka, hic okunmamis plakadan cok daha zararlidir: bir vatandasin adina yanlis
kayit uretir. Bu yuzden yapisal kural gecmezse sonuc (None, 0.0) doner.

TR plaka yapisi:  <il 01-81><1-3 harf><2-5 rakam>
Harf ve rakam sayilari birbirine baglidir:
    1 harf -> 4 veya 5 rakam      (34 A 1234)
    2 harf -> 3 veya 4 rakam      (34 AB 123)
    3 harf -> 2 veya 3 rakam      (34 ABC 12)
Bu kisit tek basina OCR'in urettigi anlamsiz dizilerin cogunu eler.
"""

from __future__ import annotations

import re

import numpy as np

# Yapisal on filtre. Harf/rakam sayisi uyumu ayrica dogrulanir (bkz. gecerli_mi).
_DESEN = re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])([A-Z]{1,3})([0-9]{2,5})$")

# Harf sayisi -> izin verilen rakam sayilari
_RAKAM_SAYISI = {1: {4, 5}, 2: {3, 4}, 3: {2, 3}}

# OCR karistirmalari. Konuma gore uygulanir: rakam bolgesinde harf->rakam,
# harf bolgesinde rakam->harf. Kor global degistirme dogru plakayi bozar.
_HARFTEN_RAKAMA = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                                 "Z": "2", "S": "5", "B": "8", "G": "6"})
_RAKAMDAN_HARFE = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"})

# Turkce harfler plakada kullanilmaz; OCR uretirse Latin karsiligina indirilir.
_TR_KATLA = str.maketrans({"Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U"})


def normalize(ham: str) -> str:
    """Bosluk/tire temizler, buyuk harfe cevirir, TR harflerini katlar."""
    return re.sub(r"[^0-9A-Z]", "", (ham or "").upper().translate(_TR_KATLA))


def gecerli_mi(plaka: str) -> bool:
    """Yapisal TR plaka kurallarina uyuyor mu."""
    m = _DESEN.fullmatch(plaka or "")
    if not m:
        return False
    harf, rakam = m.group(2), m.group(3)
    return len(rakam) in _RAKAM_SAYISI[len(harf)]


def duzelt(plaka: str) -> str:
    """Konuma duyarli OCR karistirma duzeltmesi.

    Zaten gecerli bir plakaya dokunmaz - duzeltme yalnizca son care.
    """
    p = normalize(plaka)
    if gecerli_mi(p) or len(p) < 5:
        return p

    # Il kodu: ilk 2 karakter her zaman rakamdir.
    il = p[:2].translate(_HARFTEN_RAKAMA)
    govde = p[2:]
    # Govdenin bas kismi harf, son kismi rakamdir. Harf uzunlugunu 3'ten 1'e
    # deneyip yapisal kurali saglayan ilk adayi kabul et.
    for n_harf in (3, 2, 1):
        if len(govde) <= n_harf:
            continue
        aday = (il
                + govde[:n_harf].translate(_RAKAMDAN_HARFE)
                + govde[n_harf:].translate(_HARFTEN_RAKAMA))
        if gecerli_mi(aday):
            return aday
    return p


def oku(kirpik: np.ndarray, ocr=None, min_conf: float = 0.5) -> tuple[str | None, float]:
    """Kirpilmis plaka goruntusunden plaka okur.

    `ocr` (goruntu) -> (metin, guven) imzali bir cagrilabilir. Verilmezse ANPR
    kapalidir ve (None, 0.0) doner - bu bir hata degil, yapilandirma durumudur.

    Yapisal dogrulamayi gecemeyen okuma ASLA dondurulmez.
    """
    if ocr is None or kirpik is None or kirpik.size == 0:
        return None, 0.0
    try:
        metin, conf = ocr(kirpik)
    except Exception:
        return None, 0.0
    if conf is None or conf < min_conf:
        return None, 0.0

    aday = duzelt(metin)
    if not gecerli_mi(aday):
        return None, 0.0
    return aday, float(conf)


def bicimle(plaka: str) -> str:
    """34ABC123 -> '34 ABC 123' (yalnizca goruntuleme icin)."""
    m = _DESEN.fullmatch(plaka or "")
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else plaka
