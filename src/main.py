import typer
from rich.console import Console
from sqlalchemy import text

from src.database.connection import Base, engine, get_session
from src.data_sources.football_data_co_uk import download_serie_a_range
from src.importers.serie_a_importer import import_matches_to_db
from src.features.basic_features import build_basic_features
from src.models.train_logistic import train_logistic_model
from src.models.predict_logistic import predict_match_by_teams
from src.models.evaluate_draw_rule import evaluate_draw_rule
from src.models.train_btts import train_btts_model

app = typer.Typer()
console = Console()


@app.command()
def init_db():
    """
    Create database tables.
    """
    Base.metadata.create_all(bind=engine)
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
    console.print(f"Accuracy: {result['accuracy']:.4f}")
    console.print(f"Majority baseline: {result['baseline_accuracy']:.4f}")
    console.print(f"Log loss: {result['log_loss']:.4f}")
    console.print(f"Saved to: {result['model_path']}")
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
