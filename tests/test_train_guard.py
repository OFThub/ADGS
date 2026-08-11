"""Egitim NaN muhafizi - 10+ saatlik cop kosuyu engelleyen tek kontrol.

Muhafiz gercek NaN'i beklemeden test edilir: NaN araliklidir (4 GB kartta
bellek baskisi altinda bazen cikar bazen cikmaz), bu yuzden davranis sahte bir
trainer ile deterministik olarak dogrulanir.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_yol = Path(__file__).resolve().parents[1] / "training" / "train_rdd2022.py"
_spec = importlib.util.spec_from_file_location("train_rdd2022", _yol)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _trainer(loss: float, **items: float) -> SimpleNamespace:
    return SimpleNamespace(
        loss=torch.tensor(loss),
        loss_items={k: torch.tensor(v) for k, v in items.items()},
        args=SimpleNamespace(batch=6, imgsz=640, amp=True),
    )


def test_saglikli_kayip_gecer():
    _mod._nan_muhafizi(_trainer(100.9, box_loss=2.5, cls_loss=25.0, l1_loss=0.03))


def test_toplam_kayip_nan_yakalanir():
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _mod._nan_muhafizi(_trainer(float("nan"), box_loss=2.5, cls_loss=25.0))


def test_ekrandaki_kalem_nan_yakalanir():
    # Kritik durum: toplam skaler finite ama gorunen kalem NaN. Muhafiz sadece
    # trainer.loss'a bakarsa bu vaka sessizce gecer - regresyon testi.
    with pytest.raises(RuntimeError, match="cls_loss"):
        _mod._nan_muhafizi(_trainer(100.9, box_loss=2.5, cls_loss=float("nan")))


def test_inf_de_yakalanir():
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _mod._nan_muhafizi(_trainer(100.9, box_loss=float("inf")))


def test_hata_mesaji_cozum_soyler():
    with pytest.raises(RuntimeError, match="batch<=6"):
        _mod._nan_muhafizi(_trainer(float("nan")))
