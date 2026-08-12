"""M4 ortak sozlesmesi - her ihlal dedektorunun uydugu arayuz ve geometri.

TEMEL KURAL: GEREKLI GEOMETRI YOKSA TESPIT YAPILMAZ.

Dur cizgisi tanimlanmamis bir kamerada kirmizi isik ihlali tahmin edilmez;
modul calismayi reddeder ve sebebini ctx.uyarilar'a yazar. Sessizce bos liste
donmek en tehlikeli yanlistir: kullaniciya "ihlal yok" gibi gorunur, oysa
gercek anlami "bakamadim"dir. Ayni disiplin Faz 2'de kalibrasyon icin kuruldu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from adgs.schema import Event, Party, Track, Violation

Nokta = tuple[float, float]


@dataclass
class Baglam:
    """Dedektorlerin ihtiyac duydugu her sey. Dedektorler config dosyasi okumaz."""

    video: Path
    fps: float
    source_profile: str = "cctv_fixed"
    kalib: object | None = None  # adgs.calib.Kalibrasyon
    kamera: dict = field(default_factory=dict)  # config/cameras/<id>.yaml
    kural: dict = field(default_factory=dict)  # config/rules/violations.yaml
    # Neden calisilmadigi buraya yazilir ve CLI tarafindan BASILIR.
    uyarilar: list[str] = field(default_factory=list)

    def p(self, tip: str, ad: str, varsayilan):
        """rules/violations.yaml'dan esik okur; yoksa varsayilan."""
        return (self.kural.get(tip) or {}).get(ad, varsayilan)

    def reddet(self, tip: str, sebep: str) -> list[Event]:
        """Calismayi reddet ve sebebini kaydet. Her zaman bos liste doner."""
        self.uyarilar.append(f"{tip}: {sebep} -> TESPIT YAPILMADI")
        return []

    def kalibrasyon_gecerli_mi(self, tip: str) -> bool:
        """Hiza/mesafeye bagli moduller icin on kosul."""
        if self.kalib is None:
            self.reddet(tip, "kamera kalibrasyonu verilmedi (--camera)")
            return False
        if not getattr(self.kalib, "gecerli", False):
            self.reddet(tip, "kalibrasyon dogrulamayi gecmedi (hata > esik)")
            return False
        return True


# --- Geometri ---------------------------------------------------------------


def zemin(bbox: tuple[int, int, int, int]) -> Nokta:
    """Kutunun alt orta noktasi - aracin zeminle temas ettigi yer.

    Poligon/cizgi testleri bu nokta ile yapilir: kutu merkezi kullanilirsa arac
    yuksekligi kadar sistematik hata girer ve arac, seridi/park alanini
    gercekte terk etmeden terk etmis gorunur.
    """
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def nokta_poligonda(p: Nokta, poligon: list) -> bool:
    """Ray casting. Poligon [[x,y], ...] kapali kabul edilir (son -> ilk)."""
    if not poligon or len(poligon) < 3:
        return False
    x, y = p
    icinde = False
    n = len(poligon)
    for i in range(n):
        x1, y1 = poligon[i]
        x2, y2 = poligon[(i + 1) % n]
        # Kenar y bandini kesiyor mu, kesiyorsa kesisim noktasi sagda mi
        if (y1 > y) != (y2 > y):
            kesisim_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < kesisim_x:
                icinde = not icinde
    return icinde


def _taraf(a: Nokta, b: Nokta, c: Nokta) -> float:
    """(a,b) dogrusuna gore c'nin tarafi. Isaret onemli, buyukluk degil."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def kesisiyor_mu(p1: Nokta, p2: Nokta, q1: Nokta, q2: Nokta) -> bool:
    """Iki dogru PARCASI kesisiyor mu (sonsuz dogru degil, parca).

    Dur cizgisi sonlu bir parcadir; sonsuz dogru kullanilirsa cizginin
    uzantisindan gecen, kavsaga hic girmemis araclar da "gecti" sayilir.
    """
    d1, d2 = _taraf(q1, q2, p1), _taraf(q1, q2, p2)
    d3, d4 = _taraf(p1, p2, q1), _taraf(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def kare_ciftleri(iz: Track) -> list[tuple[int, int]]:
    """Izin ardisik kare ciftleri - hareket/gecis testleri icin."""
    kareler = sorted(iz.frames)
    return list(zip(kareler, kareler[1:]))


def olay(kod: str, iz: Track, f: int, ctx: Baglam, conf: float,
         notlar: list[str], f_bas: int | None = None,
         f_son: int | None = None) -> Event:
    """Tek bir ihlali standart Event'e cevirir.

    event_id iz ve kareden turetilir: ayni ihlal iki kez uretilse bile ayni
    kimlikle gelir, rapor tekillestirmesi kolaylasir.
    """
    f_bas = f if f_bas is None else f_bas
    f_son = f if f_son is None else f_son
    kareler = {k: list(d.bbox) for k, d in iz.frames.items() if f_bas <= k <= f_son}
    return Event(
        event_id=f"{kod.lower()}_{iz.track_id}_{f}",
        tip="IHLAL",
        alt_tip=kod,
        t_start=f_bas / ctx.fps,
        t_end=f_son / ctx.fps,
        frame_start=f_bas,
        frame_end=f_son,
        source_video=str(ctx.video),
        source_profile=ctx.source_profile,
        conf=round(float(conf), 3),
        evidence={"keyframes": [f], "kareler": kareler},
        parties=[
            Party(
                track_id=iz.track_id,
                plate=iz.plate,
                # ktk_madde / kusur_sinifi / ceza_* BOS birakilir - onlari
                # Faz 5'te M6 ve M7 doldurur. M4 sadece "ne oldu"yu soyler.
                violations=[Violation(ihlal_kodu=kod, conf=round(float(conf), 3))],
            )
        ],
        notes=notlar,
    )
