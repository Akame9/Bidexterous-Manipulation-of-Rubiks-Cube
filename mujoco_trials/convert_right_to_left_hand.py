import xml.etree.ElementTree as ET

def convert_right_to_left_old(xml_in, xml_out):
    """
    Convert a right-hand MuJoCo XML into a left-hand version.
    - Renames all `robot1:` → `robotL:`
    - Updates actuator names/joints accordingly
    - Flips ctrlranges for wrist and thumb base joints
    """

    tree = ET.parse(xml_in)
    root = tree.getroot()

    # Utility: replace prefix robot1: → robotL:
    def replace_prefix(value: str) -> str:
        if value is None:
            return None
        return value.replace("robot1:", "robot1:")

    # Walk through all XML attributes
    for elem in root.iter():
        for attr in list(elem.attrib.keys()):
            elem.attrib[attr] = replace_prefix(elem.attrib[attr])

    # Special handling for actuators: adjust ctrlrange where mirroring matters
    for actuator in root.findall(".//actuator/position"):
        name = actuator.attrib.get("name", "")
        joint = actuator.attrib.get("joint", "")

        # Flip wrist joints
        if "WRJ1" in name or "WRJ1" in joint:
            lo, hi = map(float, actuator.attrib["ctrlrange"].split())
            actuator.attrib["ctrlrange"] = f"{-hi:.3f} { -lo:.3f}"

        if "WRJ0" in name or "WRJ0" in joint:
            lo, hi = map(float, actuator.attrib["ctrlrange"].split())
            actuator.attrib["ctrlrange"] = f"{-hi:.3f} { -lo:.3f}"

        # Flip thumb base (THJ0)
        if "THJ0" in name or "THJ0" in joint:
            lo, hi = map(float, actuator.attrib["ctrlrange"].split())
            actuator.attrib["ctrlrange"] = f"{-hi:.3f} { -lo:.3f}"

    # Save new XML
    tree.write(xml_out, encoding="utf-8", xml_declaration=True)
    print(f"Converted {xml_in} → {xml_out}")

# Example usage
# convert_right_to_left_old("C:\\Users\\aathi\\OneDrive - Northeastern University\\Desktop\\Lab\\MARL\\Masters_Project\\DexterousHands\\assets\\mjcf\\open_ai_assets\\hand\\shared1.xml", "shared2.xml")

import xml.etree.ElementTree as ET

def convert_right_hand_to_left(xml_in, xml_out):
    """
    Convert a right-hand MuJoCo XML into a left-hand version.
    - Renames all `robot1:` → `robotL:`
    - Flips the root body pos (mirror across X axis)
    - Flips joint axes (0 1 0 → 0 -1 0) for lateral bends
    - Flips ranges for WRJ0, WRJ1, THJ0 (wrist + thumb base)
    - Flips geom positions under palm/fingers along X
    """

    tree = ET.parse(xml_in)
    root = tree.getroot()

    def replace_prefix(val: str) -> str:
        if val is None:
            return None
        return val.replace("robot1:", "robot1:")

    # Rename everything robot1: → robotL:
    for elem in root.iter():
        for attr in elem.attrib:
            elem.attrib[attr] = replace_prefix(elem.attrib[attr])

    # Flip root hand mount position
    for body in root.findall(".//body"):
        if "hand mount" in body.attrib.get("name", ""):
            pos = list(map(float, body.attrib.get("pos", "0 0 0").split()))
            pos[0] = -pos[0]  # mirror X
            body.attrib["pos"] = " ".join(f"{p:.5f}" for p in pos)

    # Flip joint axes and ranges
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        axis = joint.attrib.get("axis")

        # Mirror Y-axis joints
        if axis:
            ax = list(map(float, axis.split()))
            if ax == [0.0, 1.0, 0.0]:
                ax[1] = -1.0
                joint.attrib["axis"] = " ".join(str(a) for a in ax)

        # Mirror ranges for wrist + thumb base
        if "WRJ1" in name or "WRJ0" in name or "THJ0" in name:
            lo, hi = map(float, joint.attrib["range"].split())
            joint.attrib["range"] = f"{-hi:.3f} { -lo:.3f}"

    # Flip geom positions under palm (X coordinate mirror)
    for geom in root.findall(".//geom"):
        if "pos" in geom.attrib:
            pos = list(map(float, geom.attrib["pos"].split()))
            pos[0] = -pos[0]
            geom.attrib["pos"] = " ".join(f"{p:.5f}" for p in pos)

    # Flip site positions too
    for site in root.findall(".//site"):
        if "pos" in site.attrib:
            pos = list(map(float, site.attrib["pos"].split()))
            pos[0] = -pos[0]
            site.attrib["pos"] = " ".join(f"{p:.5f}" for p in pos)

    # Save new XML
    tree.write(xml_out, encoding="utf-8", xml_declaration=True)
    print(f"Converted right-hand {xml_in} → left-hand {xml_out}")


# Example usage:
convert_right_hand_to_left(
    "C:\\Users\\aathi\\OneDrive - Northeastern University\\Desktop\\Lab\\MARL\\Masters_Project\\DexterousHands\\assets\\mjcf\\open_ai_assets\\hand\\robot1.xml",   # input (right-hand XML)
    "robot2.xml"    # output (left-hand XML)
)

