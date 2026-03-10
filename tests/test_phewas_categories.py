"""Unit tests for phenotype categorization."""

import pytest

from genetics_mcp_server.tools.phewas_categories import (
    categorize_phenotype,
    get_category_color,
    CATEGORY_COLORS,
)


class TestCategorizePhenotype:
    """Tests for the categorize_phenotype function."""

    def test_keyword_match_cardiovascular(self):
        """Test categorization by cardiovascular keywords."""
        assert categorize_phenotype("I9_CHD", "Coronary heart disease") == "Cardiovascular"
        assert categorize_phenotype("RANDOM", "Myocardial infarction") == "Cardiovascular"
        assert categorize_phenotype("X123", "Atrial fibrillation") == "Cardiovascular"

    def test_keyword_match_metabolic(self):
        """Test categorization by metabolic keywords."""
        assert categorize_phenotype("T2D", "Type 2 diabetes") == "Metabolic"
        assert categorize_phenotype("RANDOM", "Obesity") == "Metabolic"
        assert categorize_phenotype("X123", "High cholesterol levels") == "Metabolic"

    def test_keyword_match_neurological(self):
        """Test categorization by neurological keywords."""
        assert categorize_phenotype("AD", "Alzheimer's disease") == "Neurological"
        assert categorize_phenotype("RANDOM", "Migraine headache") == "Neurological"
        assert categorize_phenotype("X123", "Epilepsy") == "Neurological"

    def test_keyword_match_cancer(self):
        """Test categorization by cancer keywords."""
        # "Lung cancer" matches "lung" first (respiratory), so test with explicit cancer terms
        assert categorize_phenotype("RANDOM", "Melanoma") == "Cancer"
        assert categorize_phenotype("X123", "Breast carcinoma") == "Cancer"
        assert categorize_phenotype("X123", "Leukemia diagnosis") == "Cancer"

    def test_code_prefix_fallback(self):
        """Test categorization by code prefix when no name match."""
        assert categorize_phenotype("I9_UNKNOWN", None) == "Cardiovascular"
        assert categorize_phenotype("E4_UNKNOWN", "") == "Endocrine"
        assert categorize_phenotype("K11_UNKNOWN", None) == "Gastrointestinal"

    def test_known_code_exact_match(self):
        """Test categorization by known exact codes."""
        assert categorize_phenotype("T2D", None) == "Metabolic"
        assert categorize_phenotype("CAD", "") == "Cardiovascular"
        assert categorize_phenotype("COPD", None) == "Respiratory"

    def test_default_other_category(self):
        """Test fallback to 'Other' category."""
        assert categorize_phenotype("XYZ123", "Unknown thing") == "Other"
        # codes with known prefixes like "RA" will match, so use completely random codes
        assert categorize_phenotype("ZZZCODE", None) == "Other"
        assert categorize_phenotype("FOOBAR", "") == "Other"

    def test_name_priority_over_code(self):
        """Test that name keyword matching takes priority over code prefix."""
        # with cardiovascular prefix but diabetes in name, metabolic should win
        assert categorize_phenotype("I9_DIABETES", "Type 2 diabetes") == "Metabolic"
        # with gastrointestinal prefix but bone-related name, musculoskeletal should win
        assert categorize_phenotype("K11_BONE", "Bone density disorder") == "Musculoskeletal"

    def test_case_insensitive_keyword_match(self):
        """Test that keyword matching is case-insensitive."""
        assert categorize_phenotype("X", "DIABETES MELLITUS") == "Metabolic"
        assert categorize_phenotype("X", "Heart Attack") == "Cardiovascular"


class TestGetCategoryColor:
    """Tests for the get_category_color function."""

    def test_known_category_colors(self):
        """Test that known categories return their assigned colors."""
        assert get_category_color("Cardiovascular") == "#e41a1c"
        assert get_category_color("Metabolic") == "#377eb8"
        assert get_category_color("Cancer") == "#a65628"

    def test_other_category_color(self):
        """Test the 'Other' category color."""
        assert get_category_color("Other") == "#cccccc"

    def test_unknown_category_color(self):
        """Test that unknown categories return default gray."""
        assert get_category_color("NonExistentCategory") == "#cccccc"

    def test_all_defined_categories_have_colors(self):
        """Test that all defined categories have colors."""
        expected_categories = [
            "Cardiovascular", "Metabolic", "Neurological", "Respiratory",
            "Gastrointestinal", "Autoimmune", "Cancer", "Musculoskeletal",
            "Renal", "Endocrine", "Hematological", "Infectious",
            "Dermatological", "Ophthalmological", "Other",
        ]
        for category in expected_categories:
            assert category in CATEGORY_COLORS
            assert get_category_color(category) == CATEGORY_COLORS[category]
