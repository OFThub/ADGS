"""SERIT_IHLALI - VARSAYILAN OLARAK KAPALI.

Neden kapali: KTK m.55 serit degistirmeyi degil, SINYAL VERMEDEN ve guvenli
olmayan bicimde serit degistirmeyi yasaklar. Sabit CCTV 6-10 m yukseklikten
bakar; sinyal lambasinin yanip sondugunu bu mesafede guvenilir okumak pratik
degildir. Sinyal dogrulanamadigi surece her serit degisimi ihlal SAYILAMAZ -
aksi halde kurallara uygun her serit degisimi ihlal olarak kaydedilir.

Bu yuzden modul, ne oldugunu dogru soyleyecek sekilde daraltilmistir: yalnizca
ANI serit degisimleri bildirilir (N kare icinde >= K serit gecisi) ve her cikti
sinyalin dogrulanamadigini acikca tasir, guveni bilerek dusuktur. Cikti bir
ihlal tespiti degil, INCELEME ADAYIDIR.

Acmak icin: config/rules/violations.yaml -> aktif.lane: true
"""

from __future__ import annotations

from adgs.schema import Event, Track
from adgs.violations.base import Baglam, nokta_poligonda, olay, zemin

KOD = "SERIT_IHLALI"


def _serit_adi(p, seritler: list) -> str | None:
    for s in seritler:
        if nokta_poligonda(p, s.get("poligon") or []):
            return s.get("ad") or "?"
    return None


def tespit_et(izler: list[Track], ctx: Baglam) -> list[Event]:
    seritler = [s for s in (ctx.kamera.get("seritler") or []) if s.get("poligon")]
    if len(seritler) < 2:
        return ctx.reddet(
            KOD, "kamera config'inde en az 2 serit poligonu gerekir (seritler[].poligon)"
        )

    pencere = int(ctx.p(KOD, "pencere_kare", 25))
    gerekli = int(ctx.p(KOD, "min_gecis", 2))

    olaylar: list[Event] = []
    for iz in izler:
        # (kare, serit_adi) dizisi; ardisik ayni adlar teke indirilir
        dizi: list[tuple[int, str]] = []
        for f in sorted(iz.frames):
            ad = _serit_adi(zemin(iz.frames[f].bbox), seritler)
            if ad is None:
                continue
            if not dizi or dizi[-1][1] != ad:
                dizi.append((f, ad))
        gecisler = dizi[1:]  # ilk kayit "giris"tir, gecis degil
        for i in range(len(gecisler) - gerekli + 1):
            grup = gecisler[i:i + gerekli]
            f_bas, f_son = grup[0][0], grup[-1][0]
            if f_son - f_bas > pencere:
                continue
            yol = " -> ".join(ad for _f, ad in grup)
            olaylar.append(olay(
                KOD, iz, f_son, ctx,
                conf=0.3,  # bilerek dusuk - bkz. modul basligi
                notlar=[
                    f"Iz #{iz.track_id} ({iz.cls}) {f_son - f_bas} kare icinde "
                    f"{len(grup)} serit degistirdi ({yol}).",
                    "SINYAL DOGRULANAMADI: sabit CCTV mesafesinde sinyal lambasi "
                    "guvenilir okunamaz. Bu cikti bir ihlal tespiti degil, insan "
                    "incelemesi icin adaydir.",
                ],
                f_bas=f_bas, f_son=f_son,
            ))
            break  # arac basina tek kayit
    return olaylar
