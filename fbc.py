#!/usr/bin/env python3
"""
FBC Stats Tool - Analyze player statistics from FBC golf tournament data.

Commands:
    player <name>           Show stats for a specific player
    leaderboard [--by X]    Show rankings (by: points, wins, winpct, ppf)
    h2h <player1> <player2> Head-to-head record between two players
    partners <name>         Show player's performance with different partners
    courses <name>          Show player's performance at different courses
    players                 List all players
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_data(xlsx_path: str = "FBC_Data.xlsx") -> pd.DataFrame:
    """Load the Archives sheet from the Excel file."""
    path = Path(__file__).parent / xlsx_path
    df = pd.read_excel(path, sheet_name="Archives")
    return df


def normalize_name(name: str) -> str:
    """Normalize player name for matching (case-insensitive, strip whitespace)."""
    return name.strip().lower()


def find_player(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find a player by partial name match, return canonical name or None."""
    all_players = get_all_players(df)
    norm_name = normalize_name(name)

    # Exact match first
    for player in all_players:
        if normalize_name(player) == norm_name:
            return player

    # Partial match
    matches = [p for p in all_players if norm_name in normalize_name(p)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple matches for '{name}': {', '.join(matches)}")
        return None

    print(f"Player '{name}' not found. Use 'fbc.py players' to list all players.")
    return None


def get_all_players(df: pd.DataFrame) -> List[str]:
    """Get sorted list of all unique players."""
    p1 = set(df["Player 1"].dropna().unique())
    p2 = set(df["Player 2"].dropna().unique())
    # Normalize DeOteris/Deoteris
    all_players = p1 | p2
    normalized = set()
    for p in all_players:
        if p.lower() == "deoteris":
            normalized.add("DeOteris")
        else:
            normalized.add(p)
    return sorted(normalized)


def get_player_matches(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Get all matches where player participated (as Player 1 or Player 2)."""
    norm = normalize_name(player)
    mask = (df["Player 1"].apply(lambda x: normalize_name(str(x)) == norm if pd.notna(x) else False) |
            df["Player 2"].apply(lambda x: normalize_name(str(x)) == norm if pd.notna(x) else False))
    return df[mask].copy()


def cmd_players(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """List all players."""
    players = get_all_players(df)
    print(f"All Players ({len(players)}):")
    print("-" * 40)
    for i, player in enumerate(players, 1):
        print(f"  {i:2}. {player}")


def cmd_player(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Show stats for a specific player."""
    player = find_player(df, args.name)
    if not player:
        return

    matches = get_player_matches(df, player)

    if matches.empty:
        print(f"No matches found for {player}")
        return

    # Calculate stats
    wins = matches["W"].sum()
    losses = matches["L"].sum()
    ties = matches["T"].sum()
    total = wins + losses + ties
    points = matches["Points earned"].sum()
    win_pct = (wins + 0.5 * ties) / total if total > 0 else 0

    # FBC events participated
    events = sorted(matches["FBC"].dropna().unique())

    # Singles vs Doubles breakdown
    singles = matches[matches["Singes/Doubles"] == "Singles"]
    doubles = matches[matches["Singes/Doubles"] == "Doubles"]

    s_wins = singles["W"].sum() if not singles.empty else 0
    s_losses = singles["L"].sum() if not singles.empty else 0
    s_ties = singles["T"].sum() if not singles.empty else 0
    s_total = s_wins + s_losses + s_ties

    d_wins = doubles["W"].sum() if not doubles.empty else 0
    d_losses = doubles["L"].sum() if not doubles.empty else 0
    d_ties = doubles["T"].sum() if not doubles.empty else 0
    d_total = d_wins + d_losses + d_ties

    print(f"\n{'='*50}")
    print(f"  Player Stats: {player}")
    print(f"{'='*50}")
    print(f"\n  Overall Record:")
    print(f"    Wins:    {int(wins)}")
    print(f"    Losses:  {int(losses)}")
    print(f"    Ties:    {int(ties)}")
    print(f"    Total:   {int(total)} matches")
    print(f"    Win %:   {win_pct:.1%}")
    print(f"    Points:  {points:.1f}")

    print(f"\n  Singles ({int(s_total)} matches):")
    if s_total > 0:
        s_pct = (s_wins + 0.5 * s_ties) / s_total
        print(f"    {int(s_wins)}W - {int(s_losses)}L - {int(s_ties)}T ({s_pct:.1%})")
    else:
        print(f"    No singles matches")

    print(f"\n  Doubles ({int(d_total)} matches):")
    if d_total > 0:
        d_pct = (d_wins + 0.5 * d_ties) / d_total
        print(f"    {int(d_wins)}W - {int(d_losses)}L - {int(d_ties)}T ({d_pct:.1%})")
    else:
        print(f"    No doubles matches")

    print(f"\n  FBC Events: {len(events)}")
    print(f"    Participated: {', '.join(map(str, [int(e) for e in events]))}")
    print(f"    Points/Event: {points/len(events):.2f}")
    print()


def cmd_leaderboard(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Show leaderboard rankings."""
    players = get_all_players(df)
    stats = []

    for player in players:
        matches = get_player_matches(df, player)
        if matches.empty:
            continue

        wins = matches["W"].sum()
        losses = matches["L"].sum()
        ties = matches["T"].sum()
        total = wins + losses + ties
        points = matches["Points earned"].sum()
        events = len(matches["FBC"].dropna().unique())
        win_pct = (wins + 0.5 * ties) / total if total > 0 else 0
        ppf = points / events if events > 0 else 0

        stats.append({
            "Player": player,
            "W": int(wins),
            "L": int(losses),
            "T": int(ties),
            "Matches": int(total),
            "Win%": win_pct,
            "Points": points,
            "Events": events,
            "PPF": ppf
        })

    stats_df = pd.DataFrame(stats)

    # Sort by specified criteria
    sort_map = {
        "points": ("Points", False),
        "wins": ("W", False),
        "winpct": ("Win%", False),
        "ppf": ("PPF", False),
        "matches": ("Matches", False)
    }

    sort_col, ascending = sort_map.get(args.by, ("Points", False))
    stats_df = stats_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    stats_df.index = stats_df.index + 1  # 1-based ranking

    # Filter by minimum events if specified
    if args.min_events:
        stats_df = stats_df[stats_df["Events"] >= args.min_events]

    print(f"\n{'='*75}")
    print(f"  FBC Leaderboard (sorted by {args.by})")
    if args.min_events:
        print(f"  (minimum {args.min_events} events)")
    print(f"{'='*75}")
    print()

    # Format output
    print(f"  {'Rank':<5} {'Player':<15} {'W':>4} {'L':>4} {'T':>3} {'Win%':>7} {'Points':>7} {'Events':>6} {'PPF':>6}")
    print(f"  {'-'*5} {'-'*15} {'-'*4} {'-'*4} {'-'*3} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")

    for rank, row in stats_df.head(args.top).iterrows():
        print(f"  {rank:<5} {row['Player']:<15} {row['W']:>4} {row['L']:>4} {row['T']:>3} "
              f"{row['Win%']:>6.1%} {row['Points']:>7.1f} {row['Events']:>6} {row['PPF']:>6.2f}")
    print()


def cmd_h2h(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Show head-to-head record between two players."""
    player1 = find_player(df, args.player1)
    if not player1:
        return
    player2 = find_player(df, args.player2)
    if not player2:
        return

    if normalize_name(player1) == normalize_name(player2):
        print("Cannot compare a player against themselves.")
        return

    # Find matches where both players participated
    p1_matches = get_player_matches(df, player1)

    # Filter to matches involving player2 (as opponent or partner)
    norm_p2 = normalize_name(player2)

    # As opponents (player2 is Opponent1 or Opponent2)
    opponent_mask = (
        p1_matches["Opponent1"].apply(lambda x: normalize_name(str(x)) == norm_p2 if pd.notna(x) else False) |
        p1_matches["Opponent2"].apply(lambda x: normalize_name(str(x)) == norm_p2 if pd.notna(x) else False) |
        p1_matches["Singles Opponent"].apply(lambda x: normalize_name(str(x)) == norm_p2 if pd.notna(x) else False)
    )
    vs_matches = p1_matches[opponent_mask].copy()

    # As partners (player2 is Player 2 when player1 is Player 1, or vice versa)
    partner_mask = (
        p1_matches["Player 2"].apply(lambda x: normalize_name(str(x)) == norm_p2 if pd.notna(x) else False) |
        (p1_matches["Player 1"].apply(lambda x: normalize_name(str(x)) == norm_p2 if pd.notna(x) else False))
    )
    partner_matches = p1_matches[partner_mask].copy()

    print(f"\n{'='*60}")
    print(f"  Head-to-Head: {player1} vs {player2}")
    print(f"{'='*60}")

    # Opponent record
    if not vs_matches.empty:
        p1_wins = vs_matches["W"].sum()
        p1_losses = vs_matches["L"].sum()
        p1_ties = vs_matches["T"].sum()
        total = p1_wins + p1_losses + p1_ties

        print(f"\n  As Opponents ({int(total)} matches):")
        print(f"    {player1}: {int(p1_wins)}W - {int(p1_losses)}L - {int(p1_ties)}T")
        print(f"    {player2}: {int(p1_losses)}W - {int(p1_wins)}L - {int(p1_ties)}T")

        if p1_wins > p1_losses:
            print(f"\n    {player1} leads the series")
        elif p1_losses > p1_wins:
            print(f"\n    {player2} leads the series")
        else:
            print(f"\n    Series is tied")

        # Show match details
        print(f"\n  Match History (vs):")
        for _, row in vs_matches.iterrows():
            fbc = int(row["FBC"]) if pd.notna(row["FBC"]) else "?"
            course = row["Course"] if pd.notna(row["Course"]) else "?"
            result = row["W/L/T"] if pd.notna(row["W/L/T"]) else "?"
            match_type = row["Singes/Doubles"] if pd.notna(row["Singes/Doubles"]) else "?"

            # Determine partner info for doubles
            partner = ""
            if match_type == "Doubles":
                p2_col = row["Player 2"]
                if pd.notna(p2_col) and normalize_name(str(p2_col)) != normalize_name(player1):
                    partner = f" (w/ {p2_col})"

            # P1 perspective: W means p1 won
            result_str = f"{player1} {result}"
            print(f"    FBC {fbc}: {result_str} at {course}{partner} [{match_type}]")
    else:
        print(f"\n  No matches as opponents")

    # Partner record
    if not partner_matches.empty:
        wins = partner_matches["W"].sum()
        losses = partner_matches["L"].sum()
        ties = partner_matches["T"].sum()
        total = wins + losses + ties
        win_pct = (wins + 0.5 * ties) / total if total > 0 else 0

        print(f"\n  As Partners ({int(total)} matches):")
        print(f"    Record: {int(wins)}W - {int(losses)}L - {int(ties)}T ({win_pct:.1%})")
    else:
        print(f"\n  No matches as partners")

    print()


def cmd_partners(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Show player's performance with different partners."""
    player = find_player(df, args.name)
    if not player:
        return

    # Get doubles matches only
    matches = get_player_matches(df, player)
    doubles = matches[matches["Singes/Doubles"] == "Doubles"].copy()

    if doubles.empty:
        print(f"No doubles matches found for {player}")
        return

    # Find partner for each match
    norm_player = normalize_name(player)

    def get_partner(row):
        p1 = str(row["Player 1"]) if pd.notna(row["Player 1"]) else ""
        p2 = str(row["Player 2"]) if pd.notna(row["Player 2"]) else ""
        if normalize_name(p1) == norm_player:
            return p2
        else:
            return p1

    doubles["Partner"] = doubles.apply(get_partner, axis=1)

    # Group by partner
    partner_stats = []
    for partner in doubles["Partner"].unique():
        if not partner or pd.isna(partner):
            continue
        partner_matches = doubles[doubles["Partner"] == partner]
        wins = partner_matches["W"].sum()
        losses = partner_matches["L"].sum()
        ties = partner_matches["T"].sum()
        total = wins + losses + ties
        win_pct = (wins + 0.5 * ties) / total if total > 0 else 0
        points = partner_matches["Points earned"].sum()

        partner_stats.append({
            "Partner": partner,
            "W": int(wins),
            "L": int(losses),
            "T": int(ties),
            "Matches": int(total),
            "Win%": win_pct,
            "Points": points
        })

    partner_df = pd.DataFrame(partner_stats)
    partner_df = partner_df.sort_values("Matches", ascending=False)

    print(f"\n{'='*65}")
    print(f"  Partner Performance: {player}")
    print(f"{'='*65}")
    print()
    print(f"  {'Partner':<15} {'W':>4} {'L':>4} {'T':>3} {'Matches':>8} {'Win%':>7} {'Points':>7}")
    print(f"  {'-'*15} {'-'*4} {'-'*4} {'-'*3} {'-'*8} {'-'*7} {'-'*7}")

    for _, row in partner_df.iterrows():
        print(f"  {row['Partner']:<15} {row['W']:>4} {row['L']:>4} {row['T']:>3} "
              f"{row['Matches']:>8} {row['Win%']:>6.1%} {row['Points']:>7.1f}")

    # Summary
    total_matches = partner_df["Matches"].sum()
    total_wins = partner_df["W"].sum()
    total_losses = partner_df["L"].sum()
    total_ties = partner_df["T"].sum()
    overall_pct = (total_wins + 0.5 * total_ties) / total_matches if total_matches > 0 else 0

    print(f"  {'-'*15} {'-'*4} {'-'*4} {'-'*3} {'-'*8} {'-'*7} {'-'*7}")
    print(f"  {'TOTAL':<15} {total_wins:>4} {total_losses:>4} {total_ties:>3} "
          f"{total_matches:>8} {overall_pct:>6.1%} {partner_df['Points'].sum():>7.1f}")
    print()

    # Best/worst partners (min 3 matches)
    qualified = partner_df[partner_df["Matches"] >= 3].copy()
    if not qualified.empty:
        best = qualified.loc[qualified["Win%"].idxmax()]
        worst = qualified.loc[qualified["Win%"].idxmin()]
        print(f"  Best Partner (3+ matches):  {best['Partner']} ({best['Win%']:.1%} in {best['Matches']} matches)")
        print(f"  Worst Partner (3+ matches): {worst['Partner']} ({worst['Win%']:.1%} in {worst['Matches']} matches)")
        print()


def cmd_courses(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Show player's performance at different courses."""
    player = find_player(df, args.name)
    if not player:
        return

    matches = get_player_matches(df, player)

    if matches.empty:
        print(f"No matches found for {player}")
        return

    # Group by course
    course_stats = []
    for course in matches["Course"].dropna().unique():
        course_matches = matches[matches["Course"] == course]
        wins = course_matches["W"].sum()
        losses = course_matches["L"].sum()
        ties = course_matches["T"].sum()
        total = wins + losses + ties
        win_pct = (wins + 0.5 * ties) / total if total > 0 else 0
        points = course_matches["Points earned"].sum()

        course_stats.append({
            "Course": course,
            "W": int(wins),
            "L": int(losses),
            "T": int(ties),
            "Matches": int(total),
            "Win%": win_pct,
            "Points": points
        })

    course_df = pd.DataFrame(course_stats)
    course_df = course_df.sort_values("Matches", ascending=False)

    print(f"\n{'='*75}")
    print(f"  Course Performance: {player}")
    print(f"{'='*75}")
    print()
    print(f"  {'Course':<25} {'W':>4} {'L':>4} {'T':>3} {'Matches':>8} {'Win%':>7} {'Points':>7}")
    print(f"  {'-'*25} {'-'*4} {'-'*4} {'-'*3} {'-'*8} {'-'*7} {'-'*7}")

    for _, row in course_df.iterrows():
        course_name = row["Course"][:25] if len(row["Course"]) > 25 else row["Course"]
        print(f"  {course_name:<25} {row['W']:>4} {row['L']:>4} {row['T']:>3} "
              f"{row['Matches']:>8} {row['Win%']:>6.1%} {row['Points']:>7.1f}")

    # Summary
    total_matches = course_df["Matches"].sum()
    total_wins = course_df["W"].sum()
    total_losses = course_df["L"].sum()
    total_ties = course_df["T"].sum()
    overall_pct = (total_wins + 0.5 * total_ties) / total_matches if total_matches > 0 else 0

    print(f"  {'-'*25} {'-'*4} {'-'*4} {'-'*3} {'-'*8} {'-'*7} {'-'*7}")
    print(f"  {'TOTAL':<25} {total_wins:>4} {total_losses:>4} {total_ties:>3} "
          f"{total_matches:>8} {overall_pct:>6.1%} {course_df['Points'].sum():>7.1f}")
    print()

    # Best/worst courses (min 2 matches)
    qualified = course_df[course_df["Matches"] >= 2].copy()
    if not qualified.empty:
        best = qualified.loc[qualified["Win%"].idxmax()]
        worst = qualified.loc[qualified["Win%"].idxmin()]
        print(f"  Best Course (2+ matches):  {best['Course']} ({best['Win%']:.1%} in {best['Matches']} matches)")
        print(f"  Worst Course (2+ matches): {worst['Course']} ({worst['Win%']:.1%} in {worst['Matches']} matches)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="FBC Stats Tool - Analyze player statistics from FBC golf tournament data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fbc.py player hilts
  python fbc.py leaderboard --by winpct --min-events 5
  python fbc.py h2h hilts jackson
  python fbc.py partners grise
  python fbc.py courses connolly
  python fbc.py players
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # player command
    player_parser = subparsers.add_parser("player", help="Show stats for a specific player")
    player_parser.add_argument("name", help="Player name (partial match supported)")

    # leaderboard command
    lb_parser = subparsers.add_parser("leaderboard", help="Show rankings")
    lb_parser.add_argument("--by", choices=["points", "wins", "winpct", "ppf", "matches"],
                          default="points", help="Sort criteria (default: points)")
    lb_parser.add_argument("--top", type=int, default=20, help="Number of players to show (default: 20)")
    lb_parser.add_argument("--min-events", type=int, help="Minimum events to qualify")

    # h2h command
    h2h_parser = subparsers.add_parser("h2h", help="Head-to-head record between two players")
    h2h_parser.add_argument("player1", help="First player name")
    h2h_parser.add_argument("player2", help="Second player name")

    # partners command
    partners_parser = subparsers.add_parser("partners", help="Show player's partner performance")
    partners_parser.add_argument("name", help="Player name")

    # courses command
    courses_parser = subparsers.add_parser("courses", help="Show player's course performance")
    courses_parser.add_argument("name", help="Player name")

    # players command
    subparsers.add_parser("players", help="List all players")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        print("Error: FBC_Data.xlsx not found in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Dispatch to command
    commands = {
        "player": cmd_player,
        "leaderboard": cmd_leaderboard,
        "h2h": cmd_h2h,
        "partners": cmd_partners,
        "courses": cmd_courses,
        "players": cmd_players,
    }

    commands[args.command](df, args)


if __name__ == "__main__":
    main()
