# _diag/allinea_slide6_notazione.py
# Slide 6 del PPTX (Cross-Attention): porta le formule oMath alla notazione del
# beamer: Q = Z W_Q, K = X W_K, V = X W_V, un solo d_k (era L_latent, X_input, d_QKV).
# Lavora sull'XML dentro lo zip: python-pptx non enumera le formule.
#
#   python _diag/allinea_slide6_notazione.py

import os
import shutil
import zipfile

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.normpath(os.path.join(HERE, "..", "Perceiver & Perceiver IO.pptx"))
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
T = "{%s}t" % M
R = "{%s}r" % M

Z = "\U0001d44d"        # 𝑍
K_SMALL = "\U0001d458"  # 𝑘
L = "\U0001d43f"        # 𝐿
LATENT = "\U0001d459\U0001d44e\U0001d461\U0001d452\U0001d45b\U0001d461"   # 𝑙𝑎𝑡𝑒𝑛𝑡
INPUT = "\U0001d456\U0001d45b\U0001d45d\U0001d462\U0001d461"              # 𝑖𝑛𝑝𝑢𝑡
DQKV = "\U0001d444\U0001d43e\U0001d449"                                   # 𝑄𝐾𝑉


def patch(xml):
    root = etree.fromstring(xml)
    n_z = n_in = n_dk = 0
    for t in list(root.iter(T)):
        if t.text == L:
            t.text = Z; n_z += 1
        elif t.text in (LATENT, INPUT):
            r = t.getparent()
            assert etree.QName(r).localname == "r", "run inatteso"
            r.getparent().remove(r); n_in += 1
        elif t.text == DQKV:
            t.text = K_SMALL; n_dk += 1
    assert (n_z, n_in, n_dk) == (1, 3, 5), (n_z, n_in, n_dk)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def main():
    if os.path.exists(os.path.join(os.path.dirname(DECK), "~$Perceiver & Perceiver IO.pptx")):
        raise SystemExit("Il deck e' aperto in PowerPoint: chiudilo.")
    tmp = DECK + ".tmp"
    with zipfile.ZipFile(DECK) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "ppt/slides/slide6.xml":
                data = patch(data)
            zout.writestr(item, data)
    shutil.move(tmp, DECK)
    print("slide 6 aggiornata:", DECK)


if __name__ == "__main__":
    main()
