import unittest

from src.importers.openfootball_matchweek_importer import (
    _match_week,
    _name_similarity,
    _normalise_team,
    _score,
    _season_slug,
)


class OpenFootballMatchWeekTests(unittest.TestCase):
    def test_parses_matchday_number(self):
        self.assertEqual(_match_week("Matchday 38"), 38)
        self.assertEqual(_match_week("Round 7"), 7)
        self.assertIsNone(_match_week("Final"))

    def test_converts_database_season_to_openfootball_slug(self):
        self.assertEqual(_season_slug("2024/2025"), "2024-25")

    def test_supports_both_openfootball_score_formats(self):
        self.assertEqual(_score({"score": {"ft": [2, 1]}}), (2, 1))
        self.assertEqual(_score({"score": [2, 1]}), (2, 1))

    def test_normalises_common_club_name_variants(self):
        self.assertEqual(_normalise_team("FC Internazionale Milano"), "inter")
        self.assertEqual(_normalise_team("Manchester United FC"), "man united")
        self.assertEqual(_normalise_team("Athletic Club"), "ath bilbao")
        self.assertGreater(_name_similarity("Paris Saint-Germain FC", "Paris SG"), 0.8)


if __name__ == "__main__":
    unittest.main()
