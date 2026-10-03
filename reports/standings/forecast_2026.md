# 2026-27 standings projection

Roster snapshot 2026-10-03 (published rosters before opening night, 765 players; two goalies per team). Points per 82 games. Model: projected player rates × deployment-aware minutes, plus goalies, plus each player's carried share of last season's unexplained team result (docs/standings/improvements.md). **Status:** in the backtest (2021–2025) this is 14.0 points RMSE, better than the original roster model (14.6) but still not better than last season's goal differential alone (13.7); 80% ranges are ±18 points. M1 (the original roster model), M3 (blend) and B2 (last season's goal differential) are shown for reference.

| rank | team | points | 80% range | carried residual (goals) | M1 | M3 blend | B2 history | last season |
|---|---|---|---|---|---|---|---|---|
| 1 | DAL | 106 | 88–124 | +5.4 | 104 | 105 | 101 | 112 |
| 2 | TBL | 105 | 87–123 | +4.4 | 102 | 105 | 103 | 106 |
| 3 | CAR | 104 | 86–122 | +32.3 | 93 | 101 | 102 | 113 |
| 4 | EDM | 102 | 84–120 | +11.6 | 97 | 97 | 95 | 93 |
| 5 | VGK | 102 | 84–120 | +8.3 | 97 | 98 | 96 | 95 |
| 6 | COL | 101 | 83–119 | +17.2 | 96 | 109 | 111 | 121 |
| 7 | FLA | 100 | 82–118 | +2.6 | 98 | 90 | 86 | 84 |
| 8 | OTT | 98 | 80–116 | +25.8 | 90 | 95 | 97 | 99 |
| 9 | MIN | 96 | 78–114 | +2.0 | 95 | 98 | 98 | 104 |
| 10 | NYR | 95 | 77–113 | -14.1 | 100 | 93 | 88 | 77 |
| 11 | MTL | 94 | 76–112 | +2.4 | 93 | 96 | 97 | 106 |
| 12 | BUF | 94 | 76–112 | +22.2 | 87 | 96 | 100 | 109 |
| 13 | UTA | 93 | 75–111 | +10.4 | 90 | 95 | 97 | 92 |
| 14 | TOR | 93 | 75–111 | -9.8 | 95 | 86 | 83 | 78 |
| 15 | WSH | 93 | 75–111 | -2.0 | 94 | 96 | 96 | 95 |
| 16 | PHI | 93 | 75–110 | -1.9 | 94 | 93 | 91 | 98 |
| 17 | BOS | 92 | 74–110 | -3.9 | 93 | 96 | 95 | 100 |
| 18 | PIT | 91 | 73–109 | +13.1 | 87 | 94 | 98 | 98 |
| 19 | NSH | 91 | 73–109 | -9.8 | 94 | 89 | 86 | 86 |
| 20 | NJD | 89 | 71–107 | -12.4 | 93 | 88 | 86 | 87 |
| 21 | STL | 88 | 70–106 | +2.4 | 88 | 86 | 86 | 86 |
| 22 | WPG | 87 | 69–105 | -13.9 | 92 | 88 | 86 | 82 |
| 23 | LAK | 87 | 69–105 | -14.5 | 92 | 89 | 88 | 90 |
| 24 | CBJ | 87 | 69–105 | -4.0 | 89 | 89 | 90 | 92 |
| 25 | NYI | 87 | 69–104 | -12.2 | 90 | 90 | 90 | 91 |
| 26 | SJS | 85 | 68–103 | +8.5 | 84 | 81 | 83 | 86 |
| 27 | DET | 83 | 65–101 | -12.7 | 87 | 87 | 88 | 92 |
| 28 | ANA | 83 | 65–101 | +5.5 | 82 | 83 | 87 | 92 |
| 29 | SEA | 81 | 63–99 | -7.2 | 83 | 82 | 85 | 79 |
| 30 | CGY | 75 | 57–93 | -19.5 | 82 | 79 | 82 | 77 |
| 31 | CHI | 73 | 55–91 | -8.6 | 76 | 75 | 79 | 72 |
| 32 | VAN | 71 | 53–89 | -27.6 | 80 | 70 | 71 | 58 |
