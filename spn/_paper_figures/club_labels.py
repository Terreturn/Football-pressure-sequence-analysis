"""Position team abbreviations around a scatterplot."""

from __future__ import annotations

from .club_names import PLAYER_LABELS

TEAM_SHORT = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Leeds United": "Leeds",
}

NAMES = dict(
    PLAYER_LABELS,
    **{
        "Rodrigo Hernández Cascante": "Rodri",
        "Adrien Lilian Gaëtan Truffert": "Adrien Truffert",
        "Kouassi Ryan Sessegnon": "Ryan Sessegnon",
        "Carlos Noom Quomah Baleba": "Carlos Baleba",
        "Harvey Lewis Barnes": "Harvey Barnes",
        "Florentino Ibrain Morris Luís": "Florentino Luís",
        "Martín Zubimendi Ibáñez": "Martín Zubimendi",
        "Ferdi Erenay Kadıoğlu": "Ferdi Kadıoğlu",
        "André Trindade da Costa Neto": "André",
        "Mateus Gonçalo Espanha Fernandes": "Mateus Fernandes",
        "Ezri Konsa Ngoyo": "Ezri Konsa",
        "Pau Francisco Torres": "Pau Torres",
        "Cristian Gabriel Romero": "Cristian Romero",
        "Nathan Michael Collins": "Nathan Collins",
        "Murillo Santiago Costa dos Santos": "Murillo",
        "João Maria Lobo Alves Palhinha Gonçalves": "João Palhinha",
    },
)
