"""
Phenotype categorization for PheWAS plots.

Uses a multi-strategy approach to categorize phenotypes:
1. Keyword matching on phenotype names (works across data sources)
2. Code prefix matching as fallback (for common FinnGen patterns)
3. "Other" category for unmatched phenotypes
"""

# keywords to match against phenotype names (case-insensitive)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Cardiovascular": [
        "heart", "cardiac", "coronary", "artery", "arterial", "stroke",
        "hypertension", "atrial", "fibrillation", "infarction", "angina",
        "aortic", "venous", "thrombosis", "embolism", "aneurysm",
        "cardiomyopathy", "heart failure", "blood pressure",
    ],
    "Metabolic": [
        "diabetes", "glucose", "insulin", "obesity", "lipid", "cholesterol",
        "triglyceride", "bmi", "body mass", "metabolic", "glycemic",
        "hba1c", "adiposity", "waist", "weight",
    ],
    "Neurological": [
        "brain", "alzheimer", "parkinson", "epilepsy", "migraine", "dementia",
        "multiple sclerosis", "neuropathy", "seizure", "cerebral", "cognitive",
        "depression", "anxiety", "schizophrenia", "bipolar", "psychiatric",
        "mental", "mood", "psychosis",
    ],
    "Respiratory": [
        "lung", "asthma", "copd", "pulmonary", "respiratory", "bronch",
        "pneumonia", "emphysema", "airway", "breathing", "sleep apnea",
    ],
    "Gastrointestinal": [
        "crohn", "colitis", "liver", "gastric", "intestinal", "ibd",
        "bowel", "hepatic", "cirrhosis", "gallbladder", "pancrea",
        "esophag", "stomach", "colon", "digestive", "celiac",
    ],
    "Autoimmune": [
        "lupus", "psoriasis", "autoimmune", "rheumatoid", "sjogren",
        "scleroderma", "vasculitis", "inflammatory", "immune",
    ],
    "Cancer": [
        "cancer", "carcinoma", "tumor", "melanoma", "leukemia", "lymphoma",
        "neoplasm", "malignant", "oncolog", "sarcoma", "myeloma",
    ],
    "Musculoskeletal": [
        "bone", "osteoporosis", "fracture", "arthritis", "muscle",
        "joint", "spine", "spinal", "back pain", "skeletal", "cartilage",
        "tendon", "ligament", "gout",
    ],
    "Renal": [
        "kidney", "renal", "nephro", "urinary", "bladder", "urine",
        "creatinine", "glomerular", "dialysis",
    ],
    "Endocrine": [
        "thyroid", "hormone", "adrenal", "pituitary", "endocrine",
        "testosterone", "estrogen", "cortisol",
    ],
    "Hematological": [
        "blood cell", "anemia", "hemoglobin", "platelet", "leukocyte",
        "erythrocyte", "hematocrit", "coagulation", "bleeding",
    ],
    "Infectious": [
        "infection", "infectious", "sepsis", "viral", "bacterial",
        "tuberculosis", "hiv", "hepatitis", "covid", "influenza",
    ],
    "Dermatological": [
        "skin", "dermat", "eczema", "acne", "rash", "cutaneous",
    ],
    "Ophthalmological": [
        "eye", "vision", "glaucoma", "cataract", "macular", "retina",
        "optic", "blind",
    ],
}

# FinnGen-style code prefix mappings (fallback when name matching fails)
CODE_PREFIX_CATEGORIES: dict[str, str] = {
    # ICD-10 chapter-based prefixes used in FinnGen
    "I9_": "Cardiovascular",
    "I10_": "Cardiovascular",
    "E4_": "Endocrine",
    "G6_": "Neurological",
    "F5_": "Neurological",
    "J10_": "Respiratory",
    "K11_": "Gastrointestinal",
    "L12_": "Dermatological",
    "M13_": "Musculoskeletal",
    "N14_": "Renal",
    "C3_": "Cancer",
    "D3_": "Cancer",
    "H7_": "Ophthalmological",
    "H8_": "Ophthalmological",
    # common standalone codes
    "T2D": "Metabolic",
    "T1D": "Metabolic",
    "CAD": "Cardiovascular",
    "AF": "Cardiovascular",
    "HF": "Cardiovascular",
    "CKD": "Renal",
    "COPD": "Respiratory",
    "IBD": "Gastrointestinal",
    "RA": "Autoimmune",
    "SLE": "Autoimmune",
    "MS": "Neurological",
    "AD": "Neurological",
    "PD": "Neurological",
}


def categorize_phenotype(code: str, name: str | None) -> str:
    """
    Categorize a phenotype into an organ system/disease category.

    Tries multiple strategies in order:
    1. Keyword matching against the phenotype name
    2. Code prefix matching for known patterns
    3. Returns "Other" if no match found

    Args:
        code: Phenotype code (e.g., "T2D", "I9_CHD", "finngen_R10_K11_CROHN")
        name: Human-readable phenotype name, if available

    Returns:
        Category string (e.g., "Cardiovascular", "Metabolic", "Other")
    """
    # strategy 1: keyword matching on name
    if name:
        name_lower = name.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category

    # strategy 2: code prefix matching
    code_upper = code.upper()
    for prefix, category in CODE_PREFIX_CATEGORIES.items():
        if code_upper.startswith(prefix) or prefix in code_upper:
            return category

    # strategy 3: fallback
    return "Other"


# category colors for consistent plot styling
CATEGORY_COLORS: dict[str, str] = {
    "Cardiovascular": "#e41a1c",
    "Metabolic": "#377eb8",
    "Neurological": "#4daf4a",
    "Respiratory": "#984ea3",
    "Gastrointestinal": "#ff7f00",
    "Autoimmune": "#ffff33",
    "Cancer": "#a65628",
    "Musculoskeletal": "#f781bf",
    "Renal": "#999999",
    "Endocrine": "#66c2a5",
    "Hematological": "#fc8d62",
    "Infectious": "#8da0cb",
    "Dermatological": "#e78ac3",
    "Ophthalmological": "#a6d854",
    "Other": "#cccccc",
}


def get_category_color(category: str) -> str:
    """Get the color for a category, defaulting to gray for unknown categories."""
    return CATEGORY_COLORS.get(category, "#cccccc")
