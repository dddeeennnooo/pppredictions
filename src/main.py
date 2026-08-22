import typer
from collections import Counter
from datetime import date as Date
from rich.console import Console
from sqlalchemy import text

from src.database.connection import Base, engine, get_session
from src.database.migrations import ensure_schema_compatibility
from src.data_sources.football_data_co_uk import (
    download_serie_a_range,
    download_league_range,
    DEFAULT_COMPETITIONS,
)
from src.importers.serie_a_importer import import_matches_to_db
from src.importers.openfootball_matchweek_importer import (
    OPENFOOTBALL_COMPETITIONS,
    import_openfootball_match_weeks,
)
from src.importers.open_btts_data_importer import import_open_btts_data
from src.features.basic_features import build_basic_features
from src.models.train_logistic import train_logistic_model
from src.models.predict_logistic import predict_match_by_teams
from src.models.predict_upcoming import predict_upcoming_match
from src.models.evaluate_draw_rule import evaluate_draw_rule
from src.models.train_btts import train_btts_model
from src.models.train_team_scoring import train_team_scoring_models
from src.models.train_advanced_btts import train_advanced_btts_model
from src.models.train_weekly_btts import train_weekly_btts_model
from src.models.train_monthly_btts import train_monthly_btts_model
from src.models.train_dixon_coles_btts import train_dixon_coles_btts_model
from src.importers.sportmonks_importer import (
    import_sportmonks_enrichment,
    import_sportmonks_transfers,
)

app = typer.Typer()
console = Console()


@app.command()
def init_db():
    """
    Create database tables.
    """
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility()
    console.print("[green]Database initialized.[/green]")


@app.command()
def download_serie_a(
    start_year: int = 2007,
    end_year: int = 2025,
):
    """
    Download Serie A CSV files.
    Example: 2007 means season 2007/2008.
    """
    files = download_serie_a_range(start_year, end_year)
    console.print(f"[green]Downloaded {len(files)} files.[/green]")


@app.command()
def import_serie_a():
    """
    Import downloaded Serie A data into SQLite database.
    """
    count = import_matches_to_db()
    console.print(f"[green]Imported {count} matches.[/green]")


@app.command()
def download_leagues(
    start_year: int = 2019,
    end_year: int = 2025,
    competitions: str = ",".join(DEFAULT_COMPETITIONS),
):
    """Download top European leagues (I1,E0,D1,SP1,F1 by default)."""
    selected = [value.strip().upper() for value in competitions.split(",")]
    files = download_league_range(start_year, end_year, selected)
    console.print(f"[green]Downloaded {len(files)} league-season files.[/green]")


@app.command()
def import_leagues():
    """Import Serie A and all downloaded additional leagues."""
    count = import_matches_to_db(include_all_leagues=True)
    console.print(f"[green]Imported {count} new multi-league matches.[/green]")


@app.command()
def import_match_weeks(
    competitions: str = ",".join(OPENFOOTBALL_COMPETITIONS),
):
    """Fetch OpenFootball matchdays and attach them to database matches."""
    selected = {
        value.strip().upper()
        for value in competitions.split(",")
        if value.strip()
    }
    result = import_openfootball_match_weeks(selected)
    console.print("[green]OpenFootball match weeks imported.[/green]")
    console.print(f"Downloaded league-seasons: {len(result['downloaded'])}")
    console.print(f"Newly assigned matches: {result['matched']}")
    console.print(f"Already assigned matches: {result['unchanged']}")
    console.print(f"Unmatched fixtures: {len(result['unmatched'])}")
    if result["unmatched"]:
        unmatched_counts = Counter(
            (row["competition"], row["season"])
            for row in result["unmatched"]
        )
        console.print(
            "Unmatched by league-season: "
            + ", ".join(
                f"{competition} {season}={count}"
                for (competition, season), count in sorted(unmatched_counts.items())
            )
        )
        for competition_season in sorted(unmatched_counts):
            examples = [
                row
                for row in result["unmatched"]
                if (row["competition"], row["season"]) == competition_season
            ][:3]
            for row in examples:
                console.print(
                    "  {competition} {season} MW{match_week}: {date} "
                    "{home_team} vs {away_team} ({score})".format(**row)
                )
    if result["unavailable"]:
        console.print(
            "Unavailable league-seasons: " + ", ".join(result["unavailable"])
        )


@app.command()
def import_open_btts(refresh: bool = False):
    """Import open BTTS, match-stat, coach, formation and lineup data."""
    result = import_open_btts_data(refresh=refresh)
    console.print("[green]Open BTTS/xG data imported.[/green]")
    console.print(f"Source fixtures: {result['source_fixtures']}")
    console.print(f"Newly matched: {result['inserted']}")
    console.print(f"Updated: {result['updated']}")
    console.print(f"Unmatched source fixtures: {len(result['unmatched'])}")
    console.print(f"BTTS target disagreements: {result['target_disagreements']}")
    console.print(f"Matches with both xG values: {result['matched_with_xg']}")
    console.print(
        f"Matches with both shot-zone values: {result['matched_with_shot_zones']}"
    )
    console.print(
        f"Matches with both formations: {result['matched_with_lineups']}"
    )
    console.print(
        f"Matches with both confirmed starter lists: "
        f"{result['matched_with_starters']}"
    )


@app.command()
def import_sportmonks(
    start_date: str,
    end_date: str,
    league_ids: str = "",
    fetch_btts_odds: bool = True,
):
    """Import Sportmonks weather, coaches, xG, lineups, injuries and odds."""
    selected_ids = {
        int(value.strip())
        for value in league_ids.split(",")
        if value.strip()
    }
    result = import_sportmonks_enrichment(
        start_date=Date.fromisoformat(start_date),
        end_date=Date.fromisoformat(end_date),
        league_ids=selected_ids or None,
        fetch_btts_odds=fetch_btts_odds,
    )
    console.print("[green]Sportmonks enrichment imported.[/green]")
    console.print(f"New fixtures: {result['imported']}")
    console.print(f"Updated fixtures: {result['updated']}")
    console.print(f"Skipped fixtures: {result['skipped']}")
    console.print(f"BTTS odds calls: {result['odds_calls']}")


@app.command("import-sportmonks-transfers")
def import_sportmonks_transfers_command(
    start_date: str,
    end_date: str,
):
    """Import completed Sportmonks player transfers for squad-churn features."""
    result = import_sportmonks_transfers(
        start_date=Date.fromisoformat(start_date),
        end_date=Date.fromisoformat(end_date),
    )
    console.print("[green]Sportmonks transfers imported.[/green]")
    console.print(f"New team events: {result['imported']}")
    console.print(f"Skipped existing/incomplete events: {result['skipped']}")


@app.command()
def enrichment_stats():
    """Show coverage of optional provider data used by the advanced model."""
    Base.metadata.create_all(bind=engine)
    session = get_session()
    try:
        fixture_stats = session.execute(
            text(
                """
                SELECT
                    COUNT(*) AS fixtures,
                    SUM(CASE WHEN temperature_c IS NOT NULL THEN 1 ELSE 0 END) AS weather,
                    SUM(CASE WHEN home_coach_id IS NOT NULL AND away_coach_id IS NOT NULL
                             THEN 1 ELSE 0 END) AS coaches,
                    SUM(CASE WHEN home_sidelined IS NOT NULL AND away_sidelined IS NOT NULL
                             THEN 1 ELSE 0 END) AS injuries,
                    SUM(CASE WHEN home_lineup_player_ids IS NOT NULL
                                  AND away_lineup_player_ids IS NOT NULL
                             THEN 1 ELSE 0 END) AS lineups,
                    SUM(CASE WHEN home_xg IS NOT NULL AND away_xg IS NOT NULL
                             THEN 1 ELSE 0 END) AS xg,
                    SUM(CASE WHEN btts_yes_odds IS NOT NULL AND btts_no_odds IS NOT NULL
                             THEN 1 ELSE 0 END) AS btts_odds
                FROM provider_fixture_data
                """
            )
        ).mappings().one()
        transfer_events = session.execute(
            text("SELECT COUNT(*) FROM provider_team_events")
        ).scalar_one()
    finally:
        session.close()

    console.print("[cyan]Optional enrichment coverage[/cyan]")
    console.print(f"Provider fixtures: {fixture_stats['fixtures'] or 0}")
    console.print(f"Weather: {fixture_stats['weather'] or 0}")
    console.print(f"Both coaches: {fixture_stats['coaches'] or 0}")
    console.print(f"Injury/sidelined counts: {fixture_stats['injuries'] or 0}")
    console.print(f"Lineups: {fixture_stats['lineups'] or 0}")
    console.print(f"xG: {fixture_stats['xg'] or 0}")
    console.print(f"BTTS odds: {fixture_stats['btts_odds'] or 0}")
    console.print(f"Transfer team events: {transfer_events}")


@app.command()
def db_stats():
    """
    Show basic database stats.
    """
    session = get_session()

    total_matches = session.execute(
        text("SELECT COUNT(*) FROM matches")
    ).scalar()

    first_date = session.execute(
        text("SELECT MIN(date) FROM matches")
    ).scalar()

    last_date = session.execute(
        text("SELECT MAX(date) FROM matches")
    ).scalar()

    seasons = session.execute(
        text("SELECT COUNT(DISTINCT season) FROM matches")
    ).scalar()

    teams = session.execute(
        text("""
            SELECT COUNT(DISTINCT team_name) FROM (
                SELECT home_team AS team_name FROM matches
                UNION
                SELECT away_team AS team_name FROM matches
            )
        """)
    ).scalar()

    session.close()

    console.print("[cyan]Database stats[/cyan]")
    console.print(f"Matches: {total_matches}")
    console.print(f"Seasons: {seasons}")
    console.print(f"Teams: {teams}")
    console.print(f"First match date: {first_date}")
    console.print(f"Last match date: {last_date}")

@app.command()
def build_features():
    """
    Build basic pre-match features for all imported matches.
    """
    console.print("[yellow]Building features...[/yellow]")
    count = build_basic_features(clear_existing=True)
    console.print(f"[green]Built features for {count} matches.[/green]")

@app.command()
def feature_stats():
    """
    Show basic feature table stats.
    """
    session = get_session()

    total_features = session.execute(
        text("SELECT COUNT(*) FROM match_features")
    ).scalar()

    features_with_odds = session.execute(
        text("""
            SELECT COUNT(*)
            FROM match_features
            WHERE odds_home_win IS NOT NULL
              AND odds_draw IS NOT NULL
              AND odds_away_win IS NOT NULL
        """)
    ).scalar()

    first_date = session.execute(
        text("SELECT MIN(date) FROM match_features")
    ).scalar()

    last_date = session.execute(
        text("SELECT MAX(date) FROM match_features")
    ).scalar()

    session.close()

    console.print("[cyan]Feature stats[/cyan]")
    console.print(f"Feature rows: {total_features}")
    console.print(f"Rows with 1/X/2 odds: {features_with_odds}")
    console.print(f"First date: {first_date}")
    console.print(f"Last date: {last_date}")

@app.command()
def train_logistic():
    """
    Train first baseline logistic regression model for 1/X/2.
    """
    console.print("[yellow]Training logistic regression model...[/yellow]")

    result = train_logistic_model()

    console.print("[green]Model trained.[/green]")
    console.print(f"Rows total: {result['rows_total']}")
    console.print(f"Rows train: {result['rows_train']}")
    console.print(f"Rows test: {result['rows_test']}")
    console.print(f"Accuracy: {result['accuracy']:.4f}")
    console.print(f"Log loss: {result['log_loss']:.4f}")
    console.print(f"Classes: {result['classes']}")
    console.print(f"Saved to: {result['model_path']}")
    console.print("")
    console.print(result["classification_report"])


@app.command()
def train_btts():
    """Train and evaluate a both-teams-to-score logistic regression model."""
    console.print("[yellow]Training BTTS logistic regression model...[/yellow]")

    result = train_btts_model()

    console.print("[green]BTTS model trained.[/green]")
    console.print(f"Rows total: {result['rows_total']}")
    console.print(f"Rows train: {result['rows_train']}")
    console.print(f"Rows test: {result['rows_test']}")
    console.print(
        f"Test period: {result['test_start_date']} to {result['test_end_date']}"
    )
    console.print(f"Default 50% accuracy: {result['default_accuracy']:.4f}")
    console.print(
        f"40%-60% team probability rule accuracy: "
        f"{result['team_probability_accuracy']:.4f}"
    )
    console.print(
        "Rule: BTTS Yes only when both home and away implied win probabilities "
        f"are between {result['team_probability_low'] * 100:.0f}% and "
        f"{result['team_probability_high'] * 100:.0f}%"
    )
    console.print(f"Resolved predictions: {result['rows_test']}")
    console.print(f"Predicted No: {result['predicted_no_count']}")
    console.print(f"Predicted Yes: {result['predicted_yes_count']}")
    console.print(f"Majority baseline: {result['baseline_accuracy']:.4f}")
    console.print(f"Log loss: {result['log_loss']:.4f}")
    console.print(f"Saved to: {result['model_path']}")
    console.print("")
    console.print(result["classification_report"])


@app.command()
def train_team_scoring():
    """Train team-scoring models and evaluate gap/ranking BTTS rules."""
    console.print("[yellow]Training separate team-scoring models...[/yellow]")

    result = train_team_scoring_models()

    console.print("[green]Team-scoring models trained.[/green]")
    console.print(f"Rows total: {result['rows_total']}")
    console.print(f"Rows train: {result['rows_train']}")
    console.print(f"Rows test: {result['rows_test']}")
    console.print(
        f"Test period: {result['test_start_date']} to {result['test_end_date']}"
    )
    console.print(f"Home-team scoring accuracy: {result['home_score_accuracy']:.4f}")
    console.print(f"Away-team scoring accuracy: {result['away_score_accuracy']:.4f}")
    console.print(f"Home-team scoring log loss: {result['home_score_log_loss']:.4f}")
    console.print(f"Away-team scoring log loss: {result['away_score_log_loss']:.4f}")
    console.print(
        f"BTTS rules accuracy: {result['rule_btts_accuracy']:.4f}"
    )
    console.print(
        f"Average scoring-probability gap: {result['average_gap'] * 100:.2f}%"
    )
    console.print(
        f"Most-uneven 25% threshold: {result['uneven_gap_threshold'] * 100:.2f}%"
    )
    console.print(f"Most-uneven matches: {result['most_uneven_count']}")
    console.print(
        f"Below-average-gap matches: {result['below_average_gap_count']}"
    )
    console.print(
        "Bottom-five attack vs top-five defense matches: "
        f"{result['ranking_no_count']}"
    )
    console.print(f"Predicted BTTS No: {result['predicted_btts_no']}")
    console.print(f"Predicted BTTS Yes: {result['predicted_btts_yes']}")
    console.print(
        f"Home probability range: {result['home_probability_min'] * 100:.2f}% "
        f"to {result['home_probability_max'] * 100:.2f}%"
    )
    console.print(
        f"Away probability range: {result['away_probability_min'] * 100:.2f}% "
        f"to {result['away_probability_max'] * 100:.2f}%"
    )
    console.print(f"Home model saved to: {result['home_model_path']}")
    console.print(f"Away model saved to: {result['away_model_path']}")
    console.print("")
    console.print(result["classification_report"])


@app.command()
def train_advanced_btts():
    """Build advanced BTTS features, select a model, and test it chronologically."""
    console.print("[yellow]Building advanced BTTS dataset and training models...[/yellow]")
    result = train_advanced_btts_model()

    console.print("[green]Advanced BTTS model trained.[/green]")
    console.print(f"Rows total: {result['rows_total']}")
    console.print(f"Rows train: {result['rows_train']}")
    console.print(f"Rows validation: {result['rows_validation']}")
    console.print(f"Rows test: {result['rows_test']}")
    console.print(
        f"Test period: {result['test_start_date']} to {result['test_end_date']}"
    )
    console.print(f"Features: {result['feature_count']}")
    console.print("[cyan]Validation model comparison[/cyan]")
    for candidate in result["validation_results"]:
        console.print(
            f"{candidate['name']}: default={candidate['default_accuracy']:.4f}, "
            f"selected={candidate['selected_accuracy']:.4f}, "
            f"threshold={candidate['threshold']:.2f}, "
            f"features={candidate['feature_count']}"
        )
    console.print(f"Selected model: {result['selected_model']}")
    console.print(f"Selected features: {result['selected_feature_count']}")
    console.print(f"Selected threshold: {result['selected_threshold']:.2f}")
    console.print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    console.print(
        f"Competition-threshold validation accuracy: "
        f"{result['competition_validation_accuracy']:.4f}"
    )
    console.print(
        "Competition thresholds: "
        + ", ".join(
            f"{competition}={threshold:.2f}"
            for competition, threshold in result["competition_thresholds"].items()
        )
    )
    console.print(
        f"Using competition thresholds: {result['use_competition_thresholds']}"
    )
    console.print(f"Final test accuracy: {result['test_accuracy']:.4f}")
    console.print(f"Global-threshold test accuracy: {result['global_test_accuracy']:.4f}")
    console.print(
        f"Competition-threshold test accuracy: "
        f"{result['competition_test_accuracy']:.4f}"
    )
    console.print(f"Training-majority baseline: {result['baseline_accuracy']:.4f}")
    console.print(f"Always-No baseline: {result['always_no_accuracy']:.4f}")
    console.print(f"Always-Yes baseline: {result['always_yes_accuracy']:.4f}")
    console.print(f"Test BTTS rate: {result['test_btts_rate'] * 100:.2f}%")
    console.print(f"Test log loss: {result['test_log_loss']:.4f}")
    console.print(f"Predicted No: {result['predicted_no']}")
    console.print(f"Predicted Yes: {result['predicted_yes']}")
    console.print("[cyan]Confidence policy[/cyan]")
    console.print(
        f"No <= {result['confidence_no_threshold'] * 100:.0f}%, "
        f"Yes >= {result['confidence_yes_threshold'] * 100:.0f}%"
    )
    console.print(
        f"Validation: accuracy={result['confidence_validation_accuracy']:.4f}, "
        f"coverage={result['confidence_validation_coverage'] * 100:.2f}%"
    )
    console.print(
        f"Final test: accuracy={result['confidence_test_accuracy']:.4f}, "
        f"coverage={result['confidence_test_coverage'] * 100:.2f}% "
        f"({result['confidence_test_resolved']} matches)"
    )
    console.print(f"Saved to: {result['model_path']}")
    console.print("")
    console.print(result["classification_report"])


@app.command()
def train_weekly_btts(decision_policy: str = "rank"):
    """Retrain after each match week and backtest the latest season."""
    console.print("[yellow]Building week-safe features and selecting a model...[/yellow]")
    result = train_weekly_btts_model(decision_policy=decision_policy)
    console.print(
        f"[green]Weekly BTTS backtest complete for {result['test_season']}.[/green]"
    )
    console.print(
        f"Selected on {', '.join(result['calibration_seasons'])}: "
        f"{result['selected_model']} with {result['threshold_mode']} thresholds"
    )
    if result["selected_model"] == "calibrated_weekly_rank":
        console.print(
            f"Rank rule: {result['rank_probability_source']}, "
            f"top {result['rank_fraction']:.1%} per "
            f"{'competition/week' if result['rank_by_competition'] else 'week'}"
        )
    elif result["selected_model"] == "calibrated_probability_threshold":
        console.print(
            f"No-quota threshold rule: {result['rank_probability_source']}, "
            f"mode={result['threshold_mode']}"
        )
    elif result["selected_model"] == "calibrated_week_rank_schedule":
        console.print(
            "Rank rule: feature source and fraction selected separately for each "
            "week from calibration seasons"
        )
    console.print(
        f"Calibration weeks >={result['target_accuracy']:.0%}: "
        f"{result['validation_weeks_at_target']}/"
        f"{result['validation_weeks_total']}"
    )
    console.print("[cyan]Week-by-week results[/cyan]")
    for week in result["weekly_results"]:
        console.print(
            f"Week {week['match_week']:>2}: "
            f"{week['correct']:>2}/{week['matches']:<2} "
            f"accuracy={week['accuracy']:.2%}, "
            f"target={'YES' if week['target_met'] else 'NO'}, "
            f"stats={week['selected_combination']}, "
            f"policy={week['decision_policy']}, "
            f"cumulative={week['cumulative_accuracy']:.2%}, "
            f"training_rows={week['training_rows']}"
        )
    console.print(
        f"Weeks >={result['target_accuracy']:.0%}: "
        f"{result['weeks_at_target']}/{result['weeks_total']}"
    )
    console.print(
        f"Worst week accuracy: {result['worst_week_accuracy']:.2%} "
        f"(week(s) {', '.join(map(str, result['worst_weeks']))})"
    )
    console.print(f"Final accuracy: {result['test_accuracy']:.4f}")
    console.print(f"Majority baseline: {result['baseline_accuracy']:.4f}")
    console.print(f"Log loss: {result['test_log_loss']:.4f}")
    console.print(f"Brier score: {result['test_brier_score']:.4f}")
    console.print(
        f"Expected calibration error: "
        f"{result['test_calibration_error']:.4f}"
    )
    console.print(f"Predicted No: {result['predicted_no']}")
    console.print(f"Predicted Yes: {result['predicted_yes']}")
    console.print(f"Predictions: {result['predictions_path']}")
    console.print(f"Weekly summary: {result['summary_path']}")
    console.print(f"Model: {result['model_path']}")
    console.print("")
    console.print(result["classification_report"])


@app.command("train-dixon-coles-btts")
def train_dixon_coles_btts():
    """Fit and evaluate a time-decayed Dixon-Coles BTTS model."""
    console.print("[yellow]Training Dixon-Coles BTTS model...[/yellow]")
    result = train_dixon_coles_btts_model()
    console.print(
        f"[green]Dixon-Coles backtest complete for {result['test_season']}.[/green]"
    )
    console.print(
        f"Selected half-life={result['half_life_days']:.0f} days, "
        f"threshold={result['threshold']:.2f}"
    )
    console.print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    console.print(f"Test accuracy: {result['test_accuracy']:.4f}")
    console.print(f"Majority baseline: {result['baseline_accuracy']:.4f}")
    console.print(f"Log loss: {result['test_log_loss']:.4f}")
    console.print(f"Brier score: {result['test_brier_score']:.4f}")


@app.command()
def train_monthly_btts():
    """Predict each calendar month, then learn it before the next month."""
    console.print("[yellow]Building month-safe features and calibrating rules...[/yellow]")
    result = train_monthly_btts_model()
    console.print(
        f"[green]Monthly BTTS backtest complete for {result['test_season']}.[/green]"
    )
    console.print(
        f"Calibration seasons: {', '.join(result['calibration_seasons'])}"
    )
    console.print("[cyan]Month-by-month results[/cyan]")
    for month in result["monthly_results"]:
        console.print(
            f"{month['prediction_month']}: "
            f"{month['correct']:>2}/{month['matches']:<2} "
            f"accuracy={month['accuracy']:.2%}, "
            f"target={'YES' if month['target_met'] else 'NO'}, "
            f"stats={month['selected_combination']}, "
            f"top={month['rank_fraction']:.1%}, "
            f"cumulative={month['cumulative_accuracy']:.2%}, "
            f"training_rows={month['training_rows']}"
        )
    console.print(
        f"Months >={result['target_accuracy']:.0%}: "
        f"{result['months_at_target']}/{result['months_total']}"
    )
    console.print(
        f"Worst month accuracy: {result['worst_month_accuracy']:.2%} "
        f"({', '.join(result['worst_months'])})"
    )
    console.print(f"Final accuracy: {result['test_accuracy']:.4f}")
    console.print(f"Majority baseline: {result['baseline_accuracy']:.4f}")
    console.print(f"Log loss: {result['test_log_loss']:.4f}")
    console.print(f"Predicted No: {result['predicted_no']}")
    console.print(f"Predicted Yes: {result['predicted_yes']}")
    console.print(f"Predictions: {result['predictions_path']}")
    console.print(f"Monthly summary: {result['summary_path']}")
    console.print(f"Model: {result['model_path']}")
    console.print("")
    console.print(result["classification_report"])


@app.command()
def predict_match(
    home_team: str,
    away_team: str,
):
    """
    Predict probabilities for a historical team matchup.
    Example:
    python -m src.main predict-match "Inter" "Juventus"
    """
    result = predict_match_by_teams(home_team, away_team)

    probabilities = result["probabilities"]

    console.print(f"[cyan]{home_team} vs {away_team}[/cyan]")
    console.print(f"Feature source match date: {result['source_match_date']}")

    console.print(f"Home win: {probabilities.get('H', 0) * 100:.2f}%")
    console.print(f"Draw: {probabilities.get('D', 0) * 100:.2f}%")
    console.print(f"Away win: {probabilities.get('A', 0) * 100:.2f}%")    


@app.command()
def predict_upcoming(
    home_team: str,
    away_team: str,
    fixture_date: str,
    competition: str = "I1",
):
    """Predict an unplayed fixture using only completed earlier matches."""
    prediction_date = Date.fromisoformat(fixture_date)
    result = predict_upcoming_match(
        home_team=home_team,
        away_team=away_team,
        fixture_date=prediction_date,
        competition=competition,
    )
    probabilities = result["probabilities"]
    validation = result["validation_accuracy"]
    home_form = result["form"]["home"]
    away_form = result["form"]["away"]

    console.print(
        f"[cyan]{result['home_team']} vs {result['away_team']}[/cyan] "
        f"({result['fixture_date']}, {result['competition']})"
    )
    console.print(
        f"Training: {result['training_matches']} completed matches; "
        f"data through {result['data_through']}"
    )
    console.print(
        "Chronological holdout accuracy: "
        f"1/X/2 {validation['result']:.2%}, "
        f"BTTS {validation['btts']:.2%}, "
        f"O/U 2.5 {validation['over_25']:.2%}"
    )
    console.print(
        f"Recent form (last 5): {result['home_team']} "
        f"{home_form['points']:.2f} PPG, "
        f"{home_form['goals_for']:.2f} GF, {home_form['goals_against']:.2f} GA; "
        f"{result['away_team']} {away_form['points']:.2f} PPG, "
        f"{away_form['goals_for']:.2f} GF, {away_form['goals_against']:.2f} GA"
    )
    console.print("[yellow]1/X/2[/yellow]")
    console.print(f"Home win: {probabilities['home_win']:.2%}")
    console.print(f"Draw: {probabilities['draw']:.2%}")
    console.print(f"Away win: {probabilities['away_win']:.2%}")
    console.print("[yellow]Double chance[/yellow]")
    console.print(f"1X: {probabilities['home_or_draw']:.2%}")
    console.print(f"X2: {probabilities['draw_or_away']:.2%}")
    console.print(f"12: {probabilities['home_or_away']:.2%}")
    console.print("[yellow]Goals[/yellow]")
    console.print(
        f"BTTS Yes: {probabilities['btts_yes']:.2%} | "
        f"BTTS No: {probabilities['btts_no']:.2%}"
    )
    console.print(
        f"Over 2.5: {probabilities['over_25']:.2%} | "
        f"Under 2.5: {probabilities['under_25']:.2%}"
    )
    console.print(
        f"{result['home_team']} to score: {probabilities['home_scores']:.2%} | "
        f"{result['away_team']} to score: {probabilities['away_scores']:.2%}"
    )

@app.command()
def evaluate_draw_rule_command(
    max_home_away_threshold: float = 0.47,
    home_away_diff_threshold: float = 0.10,
):
    """
    Evaluate experimental draw rule on top of logistic model.
    """
    result = evaluate_draw_rule(
        max_home_away_threshold=max_home_away_threshold,
        home_away_diff_threshold=home_away_diff_threshold,
    )

    console.print("[cyan]Draw rule experiment[/cyan]")
    console.print(f"Rows test: {result['rows_test']}")
    console.print(f"Max H/A threshold: {result['max_home_away_threshold']}")
    console.print(f"H/A diff threshold: {result['home_away_diff_threshold']}")
    console.print(f"Draw predictions count: {result['draw_predictions_count']}")
    console.print("")
    console.print(f"Default accuracy: {result['default_accuracy']:.4f}")
    console.print(f"Draw rule accuracy: {result['draw_rule_accuracy']:.4f}")
    console.print("")
    console.print("[yellow]Default report[/yellow]")
    console.print(result["default_report"])
    console.print("")
    console.print("[yellow]Draw rule report[/yellow]")
    console.print(result["draw_rule_report"])    
    console.print("")
    console.print("[cyan]Draw betting simulation[/cyan]")
    console.print(f"Draw bets: {result['draw_bets_count']}")
    console.print(f"Draw won: {result['draw_bets_won']}")
    console.print(f"Draw winrate: {result['draw_bets_winrate'] * 100:.2f}%")
    console.print(f"Average draw odds: {result['draw_bets_avg_odds']:.2f}")
    console.print(f"Profit: {result['draw_bets_profit']:.2f} units")
    console.print(f"ROI: {result['draw_bets_roi'] * 100:.2f}%")

if __name__ == "__main__":
    app()
