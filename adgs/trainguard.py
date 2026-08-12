"""Egitim kayip muhafizi - 5+ saatlik cop kosuyu engelleyen tek kontrol.

Ultralytics'in kendi AMP kontrolu yalnizca cikarimi test eder; egitim sirasinda
kayip NaN'a giderse kosu sessizce devam eder ve sonunda ise yaramaz agirlik
uretir. Bu muhafiz onun yerine gecer.

Iki ayri arizayi ayirir:
    NaN/Inf       agirliklari KALICI bozar -> ANINDA durdur
    Asiri buyuk   tek sicrama olumcul degildir; ultralytics gradyani kirptigi
                  icin model hayatta kalir (gozlem: sicramadan sonra box_loss
                  dusmeye devam etti). Her sicramada iptal etmek hicbir epoch'un
                  tamamlanmasina izin vermiyordu - bu yuzden ardisik sicrama
                  sayilir.

RDD2022 ve CarDD egitimleri ayni 4 GB karti kullanir ve ayni riski tasir;
muhafizin ikinci bir kopyasi yazilmaz.
"""

from __future__ import annotations

# Saglikli baslangic degerleri: box ~2.6, cls ~29, l1 ~0.03. Bunun 300 katini
# asan bir kayip NaN kadar ise yaramazdir - gozlenen patlama 1e10 mertebesindeydi.
MAKS_KAYIP = 1e4
ARDISIK_SICRAMA_SINIRI = 5


def nan_muhafizi(trainer) -> None:
    """NaN/Inf veya absurd buyuklukte kayip goruldugunde egitimi durdurur.

    Iki yeri birden kontrol eder: trainer.loss toplam skalerdir, ekranda gorunen
    box/cls/l1 degerleri ise loss_items sozlugunden gelir - biri bozukken digeri
    saglikli olabiliyor, tek birine bakmak muhafizi kor birakir.

    Buyukluk kontrolu ayrica gerekli: fp32'de tasma NaN'a donusmez, 1e10 gibi
    sonlu ama anlamsiz bir sayi olarak kalir ve egitimi sessizce bozar.
    """
    import torch

    def _kotu(v) -> str | None:
        t = torch.as_tensor(v)
        if not torch.isfinite(t).all():
            return "NaN/Inf"
        buyuk = float(t.abs().max())
        return f"asiri buyuk ({buyuk:.3g})" if buyuk > MAKS_KAYIP else None

    bozuk: list[str] = []
    kayip = getattr(trainer, "loss", None)
    if isinstance(kayip, torch.Tensor) and (sebep := _kotu(kayip)):
        bozuk.append(f"loss: {sebep}")
    items = getattr(trainer, "loss_items", None)
    if isinstance(items, dict):
        bozuk += [f"{k}: {s}" for k, v in items.items() if (s := _kotu(v))]
    elif isinstance(items, torch.Tensor) and (sebep := _kotu(items)):
        bozuk.append(f"loss_items: {sebep}")

    olumcul = any("NaN/Inf" in b for b in bozuk)
    if not bozuk:
        trainer._adgs_sicrama = 0
        return
    if not olumcul:
        trainer._adgs_sicrama = int(getattr(trainer, "_adgs_sicrama", 0)) + 1
        if trainer._adgs_sicrama < ARDISIK_SICRAMA_SINIRI:
            return  # gecici sicrama - gradyan kirpmasi absorbe eder

    raise RuntimeError(
        f"Kayip bozuk ({'; '.join(bozuk)}) - egitim durduruldu "
        f"(batch={trainer.args.batch}, imgsz={trainer.args.imgsz}, "
        f"amp={trainer.args.amp}). Bu donanimda batch<=6 kullanin; "
        f"tekrarlarsa --batch 4."
    )
