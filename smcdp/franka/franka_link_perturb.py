"""Link-length perturbation helper for the Franka Panda URDF.

For the zero-shot link-length OOD experiment (extension to Experiment_plan §B,
i.e. eval-time-only perturbation, no retraining), we modify the joint origin
offsets in `panda.urdf` to simulate a robot whose link 3 (upper arm) or link 5
(lower arm) is slightly longer / shorter than the canonical specification.

Franka link geometry (from panda.urdf joint origins):
  panda_joint3   origin xyz="0  -0.316 0"    →  link 3 length = 0.316 m (y axis)
  panda_joint5   origin xyz="-0.0825 0.384 0" →  link 5 length = 0.384 m (y axis)

Perturbation rule:
  Δl_3 : shifts panda_joint3 origin's y component by -Δl_3 (more negative → longer)
  Δl_5 : shifts panda_joint5 origin's y component by +Δl_5 (more positive → longer)

The base manifold's stored URDF is unmodified.  We just produce a fresh URDF
string (via in-memory XML substitution), which is then fed to
`pytorch_kinematics.build_serial_chain_from_urdf` to obtain a new chain whose
forward kinematics reflects the perturbed link lengths.
"""
from __future__ import annotations

import re
from typing import Tuple


_PANDA_BASE_LINK3_Y = -0.316
_PANDA_BASE_LINK5_Y = +0.384


def make_link_perturbed_urdf(
    urdf_path: str,
    dl_3: float = 0.0,
    dl_5: float = 0.0,
) -> str:
    """Read `urdf_path`, modify panda_joint3 / panda_joint5 origins, return string.

    Δl > 0 lengthens the link, Δl < 0 shortens it.  Other joints unchanged.

    Returns the modified URDF as a string ready for
    `pytorch_kinematics.build_serial_chain_from_urdf`.
    """
    with open(urdf_path) as f:
        s = f.read()

    def _replace_origin_y(joint_name: str, new_y: float) -> None:
        nonlocal s
        # Match <joint ... name="joint_name" ...> ... <origin ... xyz="x y z" ...
        pat = re.compile(
            rf'(<joint[^>]*name="{joint_name}"[^>]*>.*?<origin[^/]*xyz=")'
            r'([^"]+)(")',
            re.DOTALL,
        )
        m = pat.search(s)
        if not m:
            raise ValueError(f"joint origin not found for {joint_name}")
        coords = m.group(2).split()
        if len(coords) != 3:
            raise ValueError(f"unexpected xyz for {joint_name}: {m.group(2)}")
        coords[1] = f"{new_y:.6f}"
        s = pat.sub(m.group(1) + " ".join(coords) + m.group(3), s, count=1)

    if abs(dl_3) > 1e-9:
        _replace_origin_y("panda_joint3", _PANDA_BASE_LINK3_Y - dl_3)
    if abs(dl_5) > 1e-9:
        _replace_origin_y("panda_joint5", _PANDA_BASE_LINK5_Y + dl_5)
    return s


def build_perturbed_chain(urdf_path: str, end_link: str,
                          dl_3: float = 0.0, dl_5: float = 0.0):
    """Convenience: return a pytorch_kinematics serial chain with perturbed links."""
    import pytorch_kinematics as pk
    import logging
    _lvl = logging.getLogger("pytorch_kinematics").level
    logging.getLogger("pytorch_kinematics").setLevel(logging.ERROR)
    try:
        urdf_str = make_link_perturbed_urdf(urdf_path, dl_3=dl_3, dl_5=dl_5)
        return pk.build_serial_chain_from_urdf(urdf_str, end_link)
    finally:
        logging.getLogger("pytorch_kinematics").setLevel(_lvl)


def link_lengths(urdf_path: str) -> Tuple[float, float]:
    """Return (link3_y, link5_y) base length values, for sanity-check / logging."""
    with open(urdf_path) as f:
        s = f.read()
    out = []
    for jn in ("panda_joint3", "panda_joint5"):
        m = re.search(
            rf'<joint[^>]*name="{jn}"[^>]*>.*?<origin[^/]*xyz="([^"]+)"',
            s, re.DOTALL)
        if not m:
            raise ValueError(jn)
        out.append(float(m.group(1).split()[1]))
    return tuple(out)
