import unittest

from mackolik_provider import _is_womens_fixture


class WomensFixtureFilterTests(unittest.TestCase):
    def test_turkish_k_marker_in_team_name(self):
        self.assertTrue(_is_womens_fixture("Arsenal (K)", "Lyon (K)", "Şampiyonlar Ligi"))

    def test_womens_competition_name(self):
        self.assertTrue(_is_womens_fixture("Arsenal", "Lyon", "UEFA Women's Champions League"))

    def test_turkish_women_label(self):
        self.assertTrue(_is_womens_fixture("Takım A", "Takım B", "Kadınlar Süper Ligi"))

    def test_regular_mens_fixture_is_kept(self):
        self.assertFalse(_is_womens_fixture("Galatasaray", "Fenerbahçe", "Süper Lig"))

    def test_regular_team_name_with_k_letter_is_not_excluded(self):
        self.assertFalse(_is_womens_fixture("Kasımpaşa", "Kayserispor", "Süper Lig"))


if __name__ == "__main__":
    unittest.main()
