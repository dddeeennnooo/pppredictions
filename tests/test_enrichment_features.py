import unittest
from datetime import date
from types import SimpleNamespace

import pandas as pd

from src.importers.sportmonks_importer import _sidelined_ids, _weather_values
from src.models.train_advanced_btts import (
    _lineup_ids,
    _recent_event_count,
    _season_context,
)


class EnrichmentFeatureTests(unittest.TestCase):
    def test_lineup_ids_parse_valid_and_invalid_json(self):
        self.assertEqual(_lineup_ids("[3, 7, 11]"), {3, 7, 11})
        self.assertEqual(_lineup_ids(None), set())
        self.assertEqual(_lineup_ids("invalid"), set())

    def test_weather_values_support_nested_provider_payload(self):
        fixture = {
            "weatherreport": {
                "temperature": {"day": 12.5},
                "humidity": "81%",
                "wind_speed": "32 kph",
                "description": "rain",
            }
        }
        values = _weather_values(fixture)
        self.assertEqual(values["temperature_c"], 12.5)
        self.assertEqual(values["humidity_percent"], 81.0)
        self.assertEqual(values["wind_speed_kph"], 32.0)
        self.assertEqual(values["weather_code"], "rain")

    def test_transfer_count_excludes_match_day_and_future_events(self):
        events = [
            SimpleNamespace(event_type="transfer_in", event_date=date(2026, 1, 11)),
            SimpleNamespace(event_type="transfer_in", event_date=date(2026, 2, 9)),
            SimpleNamespace(event_type="transfer_in", event_date=date(2026, 2, 10)),
            SimpleNamespace(event_type="transfer_out", event_date=date(2026, 2, 1)),
        ]
        count = _recent_event_count(
            events,
            pd.Timestamp("2026-02-10"),
            30,
            "transfer_in",
        )
        self.assertEqual(count, 2)

    def test_sidelined_player_ids_support_direct_and_nested_records(self):
        fixture = {
            "sidelined": [
                {"participant_id": 10, "player_id": 101},
                {"participant_id": 10, "sideline": {"player_id": 102}},
                {"participant_id": 20, "player_id": 201},
            ]
        }
        self.assertEqual(_sidelined_ids(fixture, 10), [101, 102])

    def test_new_team_is_marked_as_promoted(self):
        matches = pd.DataFrame(
            [
                {
                    "competition": "T1",
                    "season": "2024-2025",
                    "date": "2024-08-01",
                    "home_team": "Established A",
                    "away_team": "Established B",
                    "full_time_home_goals": 1,
                    "full_time_away_goals": 0,
                },
                {
                    "competition": "T1",
                    "season": "2025-2026",
                    "date": "2025-08-01",
                    "home_team": "Promoted",
                    "away_team": "Established A",
                    "full_time_home_goals": 0,
                    "full_time_away_goals": 1,
                },
            ]
        )
        context = _season_context(matches)
        previous = context["previous_season"][("T1", "2025-2026")]
        previous_teams = context["team_sets"][("T1", previous)]
        self.assertNotIn("Promoted", previous_teams)
        self.assertIn("Established A", previous_teams)


if __name__ == "__main__":
    unittest.main()
