from .labels import MATERIAL_TEXT, PART_TEXT, INTEGRITY_TEXT


def build_auto_description(fields, conf=None, thresholds=None):
    conf = conf or {}
    thresholds = thresholds or {
        "object_type": 0.0,
        "integrity": 0.0,
        "part_zone": 0.60,
        "material_group": 0.65,
    }

    tokens = [fields.get("object_type", "предмет"), INTEGRITY_TEXT.get(fields.get("integrity", "unknown"), "предмет")]

    part_zone = fields.get("part_zone", "unknown")
    if conf.get("part_zone", 1.0) >= thresholds["part_zone"] and PART_TEXT.get(part_zone):
        tokens.append(PART_TEXT[part_zone])

    material_group = fields.get("material_group", "other")
    if conf.get("material_group", 1.0) >= thresholds["material_group"] and MATERIAL_TEXT.get(material_group):
        tokens.append(MATERIAL_TEXT[material_group])

    return ", ".join(tokens)
