from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, UniqueConstraint, Index

from src.database.connection import Base


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    competition = Column(String, nullable=False, default="I1")

    season = Column(String, nullable=False)
    match_week = Column(Integer, nullable=True)
    date = Column(Date, nullable=False)

    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)

    full_time_home_goals = Column(Integer, nullable=True)
    full_time_away_goals = Column(Integer, nullable=True)
    full_time_result = Column(String, nullable=True)  # H, D, A

    half_time_home_goals = Column(Integer, nullable=True)
    half_time_away_goals = Column(Integer, nullable=True)
    half_time_result = Column(String, nullable=True)

    home_shots = Column(Integer, nullable=True)
    away_shots = Column(Integer, nullable=True)
    home_shots_on_target = Column(Integer, nullable=True)
    away_shots_on_target = Column(Integer, nullable=True)

    home_corners = Column(Integer, nullable=True)
    away_corners = Column(Integer, nullable=True)

    home_fouls = Column(Integer, nullable=True)
    away_fouls = Column(Integer, nullable=True)

    home_yellow_cards = Column(Integer, nullable=True)
    away_yellow_cards = Column(Integer, nullable=True)
    home_red_cards = Column(Integer, nullable=True)
    away_red_cards = Column(Integer, nullable=True)

    # basic odds
    odds_home_win = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    odds_away_win = Column(Float, nullable=True)

    # over/under 2.5 if available
    odds_over_25 = Column(Float, nullable=True)
    odds_under_25 = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "season",
            "date",
            "home_team",
            "away_team",
            name="uq_match_unique",
        ),
        Index(
            "ix_matches_competition_season_week",
            "competition",
            "season",
            "match_week",
        ),
    )
    
class MatchFeatures(Base):
    __tablename__ = "match_features"

    id = Column(Integer, primary_key=True, index=True)

    match_id = Column(Integer, nullable=False, unique=True)
    competition = Column(String, nullable=False, default="I1")

    season = Column(String, nullable=False)
    date = Column(Date, nullable=False)

    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)

    home_points_last_5 = Column(Float, nullable=True)
    away_points_last_5 = Column(Float, nullable=True)

    home_goals_scored_last_5 = Column(Float, nullable=True)
    away_goals_scored_last_5 = Column(Float, nullable=True)

    home_goals_conceded_last_5 = Column(Float, nullable=True)
    away_goals_conceded_last_5 = Column(Float, nullable=True)

    home_win_rate_last_5 = Column(Float, nullable=True)
    away_win_rate_last_5 = Column(Float, nullable=True)

    home_draw_rate_last_5 = Column(Float, nullable=True)
    away_draw_rate_last_5 = Column(Float, nullable=True)

    home_loss_rate_last_5 = Column(Float, nullable=True)
    away_loss_rate_last_5 = Column(Float, nullable=True)

    odds_home_win = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    odds_away_win = Column(Float, nullable=True)

    implied_probability_home = Column(Float, nullable=True)
    implied_probability_draw = Column(Float, nullable=True)
    implied_probability_away = Column(Float, nullable=True)

    bookmaker_margin = Column(Float, nullable=True)

    target_result = Column(String, nullable=True)  # H, D, A


class ProviderFixtureData(Base):
    """Optional pre-match/provider enrichment kept separate from core results."""

    __tablename__ = "provider_fixture_data"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String, nullable=False)
    provider_fixture_id = Column(String, nullable=False)
    date = Column(Date, nullable=False, index=True)
    competition_id = Column(String, nullable=True)
    season_id = Column(String, nullable=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_team_id = Column(String, nullable=True)
    away_team_id = Column(String, nullable=True)

    # Post-match xG: only rolling values from earlier fixtures may be features.
    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)

    # Pre-match market and availability information.
    btts_yes_odds = Column(Float, nullable=True)
    btts_no_odds = Column(Float, nullable=True)
    home_starters = Column(Integer, nullable=True)
    away_starters = Column(Integer, nullable=True)
    home_sidelined = Column(Integer, nullable=True)
    away_sidelined = Column(Integer, nullable=True)
    home_sidelined_player_ids = Column(Text, nullable=True)
    away_sidelined_player_ids = Column(Text, nullable=True)
    home_lineup_player_ids = Column(Text, nullable=True)
    away_lineup_player_ids = Column(Text, nullable=True)
    home_coach_id = Column(String, nullable=True)
    away_coach_id = Column(String, nullable=True)
    temperature_c = Column(Float, nullable=True)
    feels_like_c = Column(Float, nullable=True)
    humidity_percent = Column(Float, nullable=True)
    wind_speed_kph = Column(Float, nullable=True)
    precipitation_mm = Column(Float, nullable=True)
    weather_code = Column(String, nullable=True)
    fetched_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_fixture_id",
            name="uq_provider_fixture",
        ),
    )


class ProviderTeamEvent(Base):
    """Dated provider events such as player transfers, available pre-match."""

    __tablename__ = "provider_team_events"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String, nullable=False)
    provider_event_id = Column(String, nullable=False)
    event_date = Column(Date, nullable=False, index=True)
    team_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    player_id = Column(String, nullable=True)
    related_team_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    fetched_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_event_id",
            "team_id",
            "event_type",
            name="uq_provider_team_event",
        ),
    )


class OpenBttsMatchData(Base):
    """Post-match open-data facts; only lagged values may become features."""

    __tablename__ = "open_btts_match_data"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, nullable=False, unique=True, index=True)
    source_fixture_id = Column(Integer, nullable=False, unique=True)
    source_btts = Column(Integer, nullable=False)
    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)
    home_shots_inside_box = Column(Integer, nullable=True)
    away_shots_inside_box = Column(Integer, nullable=True)
    home_shots_outside_box = Column(Integer, nullable=True)
    away_shots_outside_box = Column(Integer, nullable=True)
    home_blocked_shots = Column(Integer, nullable=True)
    away_blocked_shots = Column(Integer, nullable=True)
    home_penalties = Column(Integer, nullable=True)
    away_penalties = Column(Integer, nullable=True)
    home_corners = Column(Integer, nullable=True)
    away_corners = Column(Integer, nullable=True)
    home_yellow_cards = Column(Integer, nullable=True)
    away_yellow_cards = Column(Integer, nullable=True)
    home_red_cards = Column(Integer, nullable=True)
    away_red_cards = Column(Integer, nullable=True)
    home_possession = Column(Float, nullable=True)
    away_possession = Column(Float, nullable=True)
    home_fouls = Column(Integer, nullable=True)
    away_fouls = Column(Integer, nullable=True)
    home_offsides = Column(Integer, nullable=True)
    away_offsides = Column(Integer, nullable=True)
    home_pass_accuracy = Column(Float, nullable=True)
    away_pass_accuracy = Column(Float, nullable=True)
    home_coach_name = Column(String, nullable=True)
    away_coach_name = Column(String, nullable=True)
    home_formation = Column(String, nullable=True)
    away_formation = Column(String, nullable=True)
    home_starter_ids = Column(Text, nullable=True)
    away_starter_ids = Column(Text, nullable=True)
