"""Egitim muhafizi - iraksayan kosu saatlerce bosa gitmesin.

En onemli test tloss olani: muhafiz uzun sure YALNIZCA anlik batch'e bakiyordu.
Tek bir batch patlayip sonraki normale dondugunde sayac sifirlaniyor, egitim
bozulmus agirliklarla devam ediyordu. Gercekte gorulen: box_loss 1.99 (normal)
iken cls kosu ortalamasi 4.77e+07 ve muhafiz hic tetiklenmedi.
"""

from __future__ import annotations

import pytest
import torch

from adgs.trainguard import ARDISIK_SICRAMA_SINIRI, MAKS_KAYIP, nan_muhafizi


class _Sahte:
    """Ultralytics trainer'inin muhafizin okudugu kadarini taklit eder."""

    class _Args:
        batch, imgsz, amp = 4, 640, False

    def __init__(self, loss=1.0, items=None, tloss=None):
        self.loss = torch.tensor(loss)
        self.loss_items = items if items is not None else torch.tensor([1.0, 1.0, 1.0])
        if tloss is not None:
            self.tloss = torch.tensor(tloss)
        self.args = self._Args()


def test_saglikli_kayip_gecer():
    nan_muhafizi(_Sahte())


def test_nan_hemen_durduruyor():
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        nan_muhafizi(_Sahte(loss=float("nan")))


def test_inf_hemen_durduruyor():
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        nan_muhafizi(_Sahte(items=torch.tensor([1.0, float("inf"), 1.0])))


def test_gecici_sicrama_tolere_ediliyor():
    """Tek bir buyuk deger egitimi durdurmaz - gradyan kirpmasi absorbe eder."""
    t = _Sahte(loss=MAKS_KAYIP * 10)
    for _ in range(ARDISIK_SICRAMA_SINIRI - 1):
        nan_muhafizi(t)


def test_israrli_sicrama_durduruyor():
    t = _Sahte(loss=MAKS_KAYIP * 10)
    with pytest.raises(RuntimeError, match="asiri buyuk"):
        for _ in range(ARDISIK_SICRAMA_SINIRI):
            nan_muhafizi(t)


def test_saglikli_batch_sayaci_sifirliyor():
    t = _Sahte(loss=MAKS_KAYIP * 10)
    for _ in range(ARDISIK_SICRAMA_SINIRI - 1):
        nan_muhafizi(t)
    nan_muhafizi(_Sahte())              # araya saglikli batch girerse
    t2 = _Sahte(loss=MAKS_KAYIP * 10)
    nan_muhafizi(t2)                    # sayac bastan baslar


# --- Kor nokta: kosu ortalamasi ---------------------------------------------


def test_patlamis_kosu_ortalamasi_yakalaniyor():
    """Anlik batch NORMAL, kosu ortalamasi patlamis - eski muhafiz kacirdi."""
    t = _Sahte(loss=1.99, items=torch.tensor([1.99, 2.0, 0.02]),
               tloss=[1.994, 4.768e07, 0.02295])
    with pytest.raises(RuntimeError, match="tloss"):
        for _ in range(ARDISIK_SICRAMA_SINIRI):
            nan_muhafizi(t)


def test_tloss_nan_hemen_durduruyor():
    t = _Sahte(tloss=[1.0, float("nan"), 1.0])
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        nan_muhafizi(t)


def test_saglikli_tloss_gecer():
    nan_muhafizi(_Sahte(tloss=[1.99, 2.83, 0.023]))


def test_tloss_yoksa_calisiyor():
    """Egitimin ilk batch'inde tloss henuz yok - muhafiz patlamamali."""
    nan_muhafizi(_Sahte())
