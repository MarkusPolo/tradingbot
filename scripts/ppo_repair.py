#!/usr/bin/env python3
"""
Entfernt das Präfix '_orig_mod.' aus einem SB3-Checkpoint.
Aufruf:  python ppo_repair.py ppo_rl.zip  ppo_rl_repaired.zip
"""
import sys, zipfile, tempfile, os, shutil, torch


def repair_checkpoint(src_zip: str, dst_zip: str) -> None:
    """Schreibt einen reparierten Checkpoint ohne '_orig_mod.'-Präfix."""
    with tempfile.TemporaryDirectory() as tmp:
        # Original-ZIP entpacken
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(tmp)

        # policy-Gewichte laden …
        pth = os.path.join(tmp, "policy.pth")
        sd = torch.load(pth, map_location="cpu")

        # … und Keys anpassen
        fixed = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        torch.save(fixed, pth)

        # neues ZIP schreiben
        base = os.path.splitext(dst_zip)[0]
        shutil.make_archive(base, "zip", tmp)

        # für make_archive wird automatisch „.zip“ angehängt
        if not dst_zip.endswith(".zip"):
            os.rename(base + ".zip", dst_zip)

    print("✔ Checkpoint repariert →", dst_zip)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Aufruf: python ppo_repair.py source.zip  target.zip")
    repair_checkpoint(sys.argv[1], sys.argv[2])
