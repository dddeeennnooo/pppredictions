from sqlalchemy import inspect, text

from src.database.connection import engine


def ensure_schema_compatibility() -> None:
    """Apply the small additive migration needed by existing SQLite databases."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    with engine.begin() as connection:
        if "matches" in table_names:
            columns = {column["name"] for column in inspector.get_columns("matches")}
            if "competition" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE matches ADD COLUMN competition VARCHAR "
                        "NOT NULL DEFAULT 'I1'"
                    )
                )
            if "match_week" not in columns:
                connection.execute(
                    text("ALTER TABLE matches ADD COLUMN match_week INTEGER")
                )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS "
                    "ix_matches_competition_season_week "
                    "ON matches (competition, season, match_week)"
                )
            )

        if "match_features" in table_names:
            columns = {
                column["name"]
                for column in inspector.get_columns("match_features")
            }
            if "competition" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE match_features ADD COLUMN competition VARCHAR "
                        "NOT NULL DEFAULT 'I1'"
                    )
                )

        if "provider_fixture_data" in table_names:
            columns = {
                column["name"]
                for column in inspector.get_columns("provider_fixture_data")
            }
            additions = {
                "home_coach_id": "VARCHAR",
                "away_coach_id": "VARCHAR",
                "temperature_c": "FLOAT",
                "feels_like_c": "FLOAT",
                "humidity_percent": "FLOAT",
                "wind_speed_kph": "FLOAT",
                "precipitation_mm": "FLOAT",
                "weather_code": "VARCHAR",
                "home_sidelined_player_ids": "TEXT",
                "away_sidelined_player_ids": "TEXT",
            }
            for column_name, column_type in additions.items():
                if column_name not in columns:
                    connection.execute(
                        text(
                            f"ALTER TABLE provider_fixture_data ADD COLUMN "
                            f"{column_name} {column_type}"
                        )
                    )

        if "open_btts_match_data" in table_names:
            columns = {
                column["name"]
                for column in inspector.get_columns("open_btts_match_data")
            }
            additions = {
                "home_penalties": "INTEGER",
                "away_penalties": "INTEGER",
                "home_corners": "INTEGER",
                "away_corners": "INTEGER",
                "home_yellow_cards": "INTEGER",
                "away_yellow_cards": "INTEGER",
                "home_red_cards": "INTEGER",
                "away_red_cards": "INTEGER",
                "home_possession": "FLOAT",
                "away_possession": "FLOAT",
                "home_fouls": "INTEGER",
                "away_fouls": "INTEGER",
                "home_offsides": "INTEGER",
                "away_offsides": "INTEGER",
                "home_pass_accuracy": "FLOAT",
                "away_pass_accuracy": "FLOAT",
                "home_coach_name": "VARCHAR",
                "away_coach_name": "VARCHAR",
                "home_formation": "VARCHAR",
                "away_formation": "VARCHAR",
                "home_starter_ids": "TEXT",
                "away_starter_ids": "TEXT",
            }
            for column_name, column_type in additions.items():
                if column_name not in columns:
                    connection.execute(
                        text(
                            f"ALTER TABLE open_btts_match_data ADD COLUMN "
                            f"{column_name} {column_type}"
                        )
                    )
