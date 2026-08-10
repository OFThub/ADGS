"""ADGS merkezi veri sözleşmesi.

Tüm modüller bu şemayı üretir/tüketir. Modüller birbirini doğrudan çağırmaz;
bu sayede bir modül yazılmamışken de pipeline çalışır (ilgili alanlar None kalır).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --- Sabitler ---------------------------------------------------------------

EventTip = Literal["KAZA", "IHLAL", "ALTYAPI"]
KusurSinifi = Literal["ASLI", "TALI", "TESPIT_EDILEMEDI"]
SourceProfile = Literal["cctv_fixed", "vehicle_mounted"]

# Kusur/ceza doğuran her çıktıya zorunlu olarak eklenir. Bkz. Karayolları Trafik
# Yönetmeliği m.156/3 — tutanak kusur oranı belirtmez; oranı TRAMER belirler.
ON_DEGERLENDIRME_NOTU = (
    "Bu çıktı görüntü analizine dayalı bir ön değerlendirmedir. Kusur oranı; "
    "kaza tespit tutanağı, tanık ifadeleri, bilirkişi/eksper raporu ve TRAMER "
    "değerlendirmesi ile yetkili merciler tarafından belirlenir. Bu sistem "
    "bağlayıcı bir tespit üretmez."
)

# Notun eklendiği olay tipleri. ALTYAPI belediyenin kendi görev alanı olduğu
# için kusur/ceza notu taşımaz.
_NOT_GEREKTIREN_TIPLER = ("KAZA", "IHLAL")


# --- Tespit / takip ---------------------------------------------------------


@dataclass
class Detection:
    """Tek karedeki tek nesne."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    cls: str
    conf: float


@dataclass
class Track:
    """M2 çıktısı — kareler boyunca kalıcı ID'li nesne.

    `world_xy` yalnızca kamera kalibrasyonu doğrulamayı geçtiyse doldurulur.
    None olması hız/mesafe modüllerinin çalışmaması gerektiği anlamına gelir.
    """

    track_id: int
    cls: str
    frames: dict[int, Detection] = field(default_factory=dict)
    plate: str | None = None
    plate_conf: float = 0.0
    world_xy: dict[int, tuple[float, float]] | None = None


# --- Olay ------------------------------------------------------------------


@dataclass
class Violation:
    """Bir ihlal. M4 üretir, M6 hukuki alanları, M7 ceza alanlarını doldurur."""

    ihlal_kodu: str  # KIRMIZI_ISIK, TERS_YON, ...
    conf: float
    ktk_madde: str | None = None  # M6
    kusur_sinifi: KusurSinifi | None = None  # M6
    ceza_tutari_try: int | None = None  # M7
    ceza_puani: int | None = None  # M7
    ceza_tablosu_tarihi: str | None = None  # M7 — ISO YYYY-MM-DD


@dataclass
class Party:
    """Olaya taraf olan bir nesne (araç/yaya)."""

    track_id: int
    plate: str | None = None
    violations: list[Violation] = field(default_factory=list)
    hasar: dict | None = None  # M8


@dataclass
class Event:
    """Sistemin tek merkezi sözleşmesi.

    Kanıtsız Event rapora yazılmaz: `evidence` en az bir keyframe veya klip
    taşımalıdır. KAZA/IHLAL olaylarında ön-değerlendirme notu otomatik eklenir.
    """

    event_id: str
    tip: EventTip
    alt_tip: str
    t_start: float  # saniye
    t_end: float
    frame_start: int
    frame_end: int
    source_video: str
    source_profile: SourceProfile
    conf: float
    evidence: dict = field(default_factory=dict)  # {"clip": str, "keyframes": [int]}
    gps: tuple[float, float] | None = None  # (lat, lon)
    parties: list[Party] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tip in _NOT_GEREKTIREN_TIPLER and ON_DEGERLENDIRME_NOTU not in self.notes:
            self.notes.append(ON_DEGERLENDIRME_NOTU)

    def kanitli_mi(self) -> bool:
        """Kanıt taşıyıp taşımadığı. Rapora yazılmadan önce kontrol edilir."""
        return bool(self.evidence.get("clip") or self.evidence.get("keyframes"))
