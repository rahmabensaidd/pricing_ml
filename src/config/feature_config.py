# ==============================================
# GLOBAL MODEL TRAINING COLUMNS
# ==============================================

# In the model, booleans are in NUM_COLS and treated as numeric (0/1)
NUM_COLS = [
    "quantity",
    "production_page",
    "height",
    "thickness",
    "width",
    # Booleans treated as numeric (0/1)
    "security_label",
    "has_coil",
    "has_insert",
    "has_tab",
    "has_backcover",
    "perf",
    "double_sided_cover",
    "shrinkwrap",
    "three_hole_drill"
]

CAT_COLS = [
    "text_paper_type",
    "text_color",
    "cover_finish_type",
    "cover_color",
    "cover_size",
    "cover_paper_type",
    "head_and_tail",
    "priority_level",
    "binding_type",
    "coil_type",
    "tab_color",
    "insert_paper_type",
    "case_finish_type",
    "spine_type",
    "label_type",
    "siren"
]

ALL_FEATURES = NUM_COLS + CAT_COLS

# Types for conversions
INT_COLS = [
    "quantity",
    "production_page",
    "security_label",
    "has_coil",
    "has_insert",
    "has_tab",
    "has_backcover",
    "perf",
    "double_sided_cover",
    "shrinkwrap",
    "three_hole_drill"
]

FLOAT_COLS = ["height", "thickness", "width"]

# ==============================================
# KNOWN CATEGORIES VOCABULARY
# ==============================================

KNOWN_CATEGORIES = {
    "text_paper_type": ["BIRCH_W40_TB", "80_GLOSS_TEXT", "70_OFFSET", "NONE", "missing", "unknown"],
    "text_color": ["4/4", "4/0", "1/1", "NONE", "missing", "unknown"],
    "cover_finish_type": ["LAYFLAT-GLOSS", "GLOSS", "MATTE", "NONE", "missing", "unknown"],
    "cover_color": ["4/0", "4/4", "1/0", "NONE", "missing", "unknown"],
    "cover_size": ["L", "M", "S", "XL", "NONE", "missing", "unknown"],
    "cover_paper_type": ["12PT_C1S", "100_GLOSS_TEXT", "80_GLOSS_TEXT", "NONE", "missing", "unknown"],
    "head_and_tail": ["NONE", "BLACK", "WHITE", "BLACK & WHITE", "missing", "unknown"],
    "priority_level": ["NORMAL", "HIGH1", "HIGH2", "LOW", "missing", "unknown"],
    "binding_type": ["SS", "CASEBIND", "PERFECT", "SPIRAL", "missing", "unknown"],
    "coil_type": ["NONE", "METAL", "PLASTIC", "missing", "unknown"],
    "tab_color": ["NONE", "WHITE", "COLOR", "missing", "unknown"],
    "insert_paper_type": ["NONE", "80_GLOSS_TEXT", "70_OFFSET", "missing", "unknown"],
    "case_finish_type": ["NONE", "LAYFLAT-GLOSS", "GLOSS", "missing", "unknown"],
    "spine_type": ["NONE", "ROUND", "SQUARE", "missing", "unknown"],
    "label_type": ["NONE", "STANDARD", "CUSTOM", "missing", "unknown"],
    "siren": ["SAV", "missing", "unknown"]
}

FALLBACK_CATEGORY = "missing"