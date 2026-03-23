#!/usr/bin/env python3
import os
import math
import argparse
import xml.etree.ElementTree as ET

# ---------------- config ----------------
# Force this exact meshdir in the top <compiler />
MESH_DIR_TARGET = "asset/smpl_meshes/geom"

# ---------------- helpers ----------------
def deg2rad_pair(r):
    try:
        lo, hi = [float(x) for x in r.strip().split()]
        return f"{lo * math.pi/180.0:.6f} {hi * math.pi/180.0:.6f}"
    except Exception:
        return r

def find_or_create(parent, tag):
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    return el

def rewrite_mesh_files_to_basenames(asset):
    if asset is None:
        return
    for m in asset.findall("mesh"):
        f = m.get("file")
        if f:
            m.set("file", os.path.basename(f))  # strip any directories

def ensure_top_compiler_with_forced_meshdir(root):
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.set("coordinate", "local")
    comp.set("meshdir", MESH_DIR_TARGET)

def ensure_mesh_geom_contact_attrs(geom):
    if geom.get("type") == "mesh":
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("condim", "3")
        geom.set("margin", "0.001")

def update_joint_attributes(j):
    r = j.get("range")
    if r is not None:
        j.set("range", deg2rad_pair(r))
        j.set("limited", "true")
    j.set("armature", "0.1")
    if j.get("damping") is None:
        j.set("damping", "0.01")
    j.set("stiffness", "10")

def remove_world_floor_and_light(worldbody):
    to_remove = []
    for elem in list(worldbody):
        if elem.tag == "geom" and elem.get("type") == "plane":
            to_remove.append(elem)
        elif elem.tag == "light":
            to_remove.append(elem)
    for e in to_remove:
        worldbody.remove(e)

def set_pelvis_root(worldbody):
    pelvis = None
    for b in worldbody.findall("body"):
        if b.get("name") == "Pelvis":
            pelvis = b
            break
    if pelvis is None:
        return
    pelvis.set("pos", "0 0 1.0")
    if pelvis.find("freejoint") is None:
        fj = ET.SubElement(pelvis, "freejoint")
        fj.set("name", "Pelvis")

def apply_to_all_joints(elem):
    for j in elem.iter("joint"):
        update_joint_attributes(j)

def ensure_mesh_geom_contacts_everywhere(root):
    for g in root.iter("geom"):
        ensure_mesh_geom_contact_attrs(g)

def add_model2_defaults(root):
    dflt = root.find("default")
    if dflt is None:
        dflt = ET.SubElement(root, "default")

    j1 = ET.SubElement(dflt, "joint")
    j1.set("damping", "0.01")
    j1.set("armature", "0.1")
    j1.set("stiffness", "10.0")
    j1.set("limited", "true")

    g = ET.SubElement(dflt, "geom")
    g.set("conaffinity", "1")
    g.set("condim", "3")
    g.set("contype", "7")
    g.set("margin", "0.001")
    g.set("rgba", "0.8 0.6 .4 1")
    g.set("type", "box")
    g.set("size", "0.01 0.01 0.01")

    j2 = ET.SubElement(dflt, "joint")
    j2.set("damping", "1")
    j2.set("armature", "0.1")

def add_tail_compiler_and_stat(root):
    tail_comp = ET.Element("compiler")
    tail_comp.set("angle", "radian")
    tail_comp.set("autolimits", "true")
    root.append(tail_comp)
    stat = ET.Element("statistic")
    stat.set("meansize", "0.05")
    root.append(stat)

def add_ctrlrange_to_motors(root, ctrlrange="-200 200"):
    act = root.find("actuator")
    if act is None:
        return
    for m in act.findall("motor"):
        m.set("ctrlrange", ctrlrange)

def pretty_write(tree, out_path):
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Convert MuJoCo XML (style 1) to style 2.")
    ap.add_argument("input_xml", help="Path to input model-1 XML")
    ap.add_argument("output_xml", help="Path to write model-2-like XML")
    args = ap.parse_args()

    tree = ET.parse(args.input_xml)
    root = tree.getroot()

    # Force meshdir and strip mesh file paths to basenames
    rewrite_mesh_files_to_basenames(root.find("asset"))
    ensure_top_compiler_with_forced_meshdir(root)

    # Defaults like model (2)
    add_model2_defaults(root)

    # World edits
    worldbody = root.find("worldbody")
    if worldbody is not None:
        remove_world_floor_and_light(worldbody)
        set_pelvis_root(worldbody)

    # Mesh geoms + joints
    ensure_mesh_geom_contacts_everywhere(root)
    apply_to_all_joints(root)

    # Actuators ctrlrange like model (2)
    add_ctrlrange_to_motors(root, ctrlrange="-200 200")

    # Trailing compiler/statistic
    add_tail_compiler_and_stat(root)

    pretty_write(tree, args.output_xml)
    print(f"Wrote converted XML to: {args.output_xml}")

if __name__ == "__main__":
    main()
