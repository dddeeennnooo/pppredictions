from sqlalchemy import Column, Integer, String, Float, Date, UniqueConstraint

from src.database.connection import Base


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)

    season = Column(String, nullable=False)
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
    )
    
class MatchFeatures(Base):
    __tablename__ = "match_features"

    id = Column(Integer, primary_key=True, index=True)

    match_id = Column(Integer, nullable=False, unique=True)

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