-- PM-EPA relational schema (DDL). The pipeline stores these tables as Parquet under
-- data/processed/ and data/outputs/; this file documents types, keys and relationships so the
-- data can be loaded into any SQL engine (DuckDB can query the Parquet files directly:
--   SELECT * FROM 'data/processed/events.parquet' LIMIT 5;).
-- Coordinates are feet, rotated so the possessing/acting team attacks +x (goal line x = +89).

CREATE TABLE games (
    game_id          BIGINT PRIMARY KEY,          -- NHL game id, e.g. 2024020001
    season           INTEGER NOT NULL,            -- start year: 2024 = 2024-25
    game_date        DATE NOT NULL,
    home_id          INTEGER NOT NULL,
    away_id          INTEGER NOT NULL,
    home_abbrev      VARCHAR(3) NOT NULL,
    away_abbrev      VARCHAR(3) NOT NULL,
    home_final       INTEGER,
    away_final       INTEGER,
    last_period_type VARCHAR(3)                   -- REG / OT / SO
);

CREATE TABLE players (
    player_id        BIGINT PRIMARY KEY,
    first_name       VARCHAR NOT NULL,
    last_name        VARCHAR NOT NULL,
    position         VARCHAR(1) NOT NULL,         -- C / L / R / D / G
    shoots           VARCHAR(1),                  -- L / R (roster handedness)
    roster_position  VARCHAR(1)
);

CREATE TABLE events (
    game_id          BIGINT NOT NULL REFERENCES games(game_id),
    event_idx        INTEGER NOT NULL,            -- 0-based order within game (sortOrder rank)
    event_id         INTEGER NOT NULL,            -- NHL eventId
    season           INTEGER NOT NULL,
    period           INTEGER NOT NULL,
    period_type      VARCHAR(3) NOT NULL,         -- REG / OT (shootouts dropped)
    t_period         DOUBLE NOT NULL,             -- seconds elapsed in period
    t_game           DOUBLE NOT NULL,             -- (period-1)*1200 + t_period
    type             VARCHAR NOT NULL,            -- faceoff, hit, giveaway, takeaway, shot-on-goal, missed-shot, blocked-shot, goal, penalty, stoppage, ...
    team_id          INTEGER,                     -- acting team (winner / hitter / giver / taker / shooter / penalized)
    opp_team_id      INTEGER,
    is_home          BOOLEAN,
    player1          BIGINT REFERENCES players(player_id),  -- primary actor
    player2          BIGINT REFERENCES players(player_id),  -- loser / hittee / blocker / goalie / drawn-by
    assist1          BIGINT,
    assist2          BIGINT,
    x_raw DOUBLE, y_raw DOUBLE,                   -- as published
    x DOUBLE, y DOUBLE,                           -- normalized to the acting team's attacking frame
    zone_code_raw    VARCHAR(1),
    shot_type        VARCHAR,
    reason           VARCHAR,                     -- stoppage reason / miss reason / penalty descKey
    home_skaters INTEGER, away_skaters INTEGER, home_goalie INTEGER, away_goalie INTEGER,
    own_skaters INTEGER, opp_skaters INTEGER, own_goalie INTEGER, opp_goalie INTEGER,
    strength         VARCHAR,                     -- 5v5 / PP / PK / 4v4 / 3v3 / EN / other (acting team)
    home_score INTEGER, away_score INTEGER,       -- before the event
    score_diff       INTEGER,                     -- acting team minus opponent, before the event
    PRIMARY KEY (game_id, event_idx)
);

CREATE TABLE shifts (
    game_id   BIGINT NOT NULL REFERENCES games(game_id),
    player_id BIGINT NOT NULL REFERENCES players(player_id),
    team_id   INTEGER NOT NULL,
    period    INTEGER NOT NULL,
    start     DOUBLE NOT NULL,                    -- seconds into period
    "end"     DOUBLE NOT NULL
);

CREATE TABLE onice (                              -- skaters on ice at each event (goalies excluded)
    game_id   BIGINT NOT NULL,
    event_idx INTEGER NOT NULL,
    home_on   BIGINT[] NOT NULL,
    away_on   BIGINT[] NOT NULL,
    PRIMARY KEY (game_id, event_idx),
    FOREIGN KEY (game_id, event_idx) REFERENCES events(game_id, event_idx)
);

CREATE TABLE shots_xg (
    game_id BIGINT NOT NULL, event_idx INTEGER NOT NULL,
    season INTEGER, type VARCHAR, team_id INTEGER, player1 BIGINT,
    x DOUBLE, y DOUBLE, dist DOUBLE, angle DOUBLE, shot_type_c VARCHAR,
    rebound BOOLEAN, rush BOOLEAN, strength VARCHAR, empty_net BOOLEAN,
    goal SMALLINT NOT NULL,
    xg DOUBLE NOT NULL,                           -- cross-fitted by season; 0 for blocked attempts
    PRIMARY KEY (game_id, event_idx),
    FOREIGN KEY (game_id, event_idx) REFERENCES events(game_id, event_idx)
);

CREATE TABLE possessions (
    poss_id            BIGINT PRIMARY KEY,        -- game_id * 10000 + local index
    game_id            BIGINT NOT NULL REFERENCES games(game_id),
    season             INTEGER NOT NULL,
    period             INTEGER NOT NULL,
    team_id            INTEGER NOT NULL,          -- possessing team
    start_type         VARCHAR NOT NULL,          -- faceoff / takeaway / loose_* / rebound / unobserved
    start_link         BIGINT REFERENCES possessions(poss_id),  -- possession whose end created this start
    end_type           VARCHAR NOT NULL,          -- shot_sog / shot_miss / shot_block / goal / giveaway / takeaway_against / unobserved_loss / stoppage:<reason> / penalty / period_end
    terminal_event_idx INTEGER,                   -- the terminal event, if the possession's own
    end_event_idx      INTEGER,                   -- the event that revealed the end
    start_t DOUBLE, end_t DOUBLE,
    n_obs              INTEGER NOT NULL,
    G                  DOUBLE,                    -- terminal xG (possessions_valued)
    goal               SMALLINT,
    end_class          SMALLINT,                  -- 1 scoring shot, 2 other end, 0 censored
    excluded           BOOLEAN                    -- empty net / unclassifiable strength (+ takeaway-linked)
);

CREATE TABLE observations (
    poss_id     BIGINT NOT NULL REFERENCES possessions(poss_id),
    seq         INTEGER NOT NULL,                 -- order within possession
    game_id BIGINT, season INTEGER, period INTEGER, team_id INTEGER,
    event_idx   INTEGER NOT NULL,                 -- source event (the origin event for inferred starts)
    t           DOUBLE NOT NULL,
    obs_kind    VARCHAR NOT NULL,                 -- faceoff / takeaway / loose_* / rebound / hit_mid / shot / giveaway / lost_takeaway
    role        VARCHAR NOT NULL,                 -- start / mid / end
    player_id   BIGINT,                           -- player on puck, NULL if unknown
    x DOUBLE, y DOUBLE,                           -- possessing team's frame
    strength VARCHAR, score_diff INTEGER,
    v DOUBLE, v_att DOUBLE, v_cf DOUBLE, v_cf_unobs DOUBLE,   -- obs_valued.parquet
    PRIMARY KEY (poss_id, seq)
);

CREATE TABLE transitions (                        -- one row per movement segment (schemas/pm_epa_transitions.parquet)
    game_id BIGINT, season INTEGER,
    poss_id BIGINT NOT NULL REFERENCES possessions(poss_id),
    team_id INTEGER, seq INTEGER NOT NULL, strength VARCHAR,
    from_kind VARCHAR, to_kind VARCHAR,
    x DOUBLE, y DOUBLE, to_x DOUBLE, to_y DOUBLE,
    start_player BIGINT, end_player BIGINT,
    gap DOUBLE,                                   -- seconds between the observations
    v_before DOUBLE, v_after DOUBLE, delta DOUBLE,
    seg_type VARCHAR,                             -- carry / split / long_gap / rebound
    w_start DOUBLE, w_end DOUBLE, w_pool DOUBLE,
    event_idx INTEGER,                            -- end event
    PRIMARY KEY (poss_id, seq)
);

CREATE TABLE credits (                            -- one row per (credit, player); conserves value
    game_id BIGINT, season INTEGER,
    poss_id BIGINT REFERENCES possessions(poss_id),   -- NULL for orphan takeaway debits
    team_id INTEGER, strength VARCHAR, event_idx INTEGER,
    player_id BIGINT REFERENCES players(player_id),   -- NULL = team-level / unattributed
    component VARCHAR NOT NULL,                   -- carry, move_start, move_end, move_pool, takeaway, takeaway_against, giveaway, shot_attempt, unobserved_loss, stoppage_loss, shot_quality, unattributed
    value DOUBLE NOT NULL                         -- expected goals
);

CREATE TABLE player_season (                      -- data/outputs/pm_epa_player_leaderboards.csv (+ extras)
    player_id BIGINT NOT NULL, season INTEGER NOT NULL,
    player_name VARCHAR, team VARCHAR, position VARCHAR(1), pos_group VARCHAR(1),
    gp INTEGER, toi_total DOUBLE, toi_ev DOUBLE, toi_pp DOUBLE, toi_pk DOUBLE,
    pm_epa_total DOUBLE, pm_epa_ev DOUBLE, pm_epa_pp DOUBLE, pm_epa_pk DOUBLE,
    pm_epa_per60_total DOUBLE, pm_epa_per60_ev DOUBLE, pm_epa_per60_pp DOUBLE, pm_epa_per60_pk DOUBLE,
    pm_epa_per100_poss DOUBLE, n_plays INTEGER, n_possessions_involved INTEGER, plays_pos INTEGER, plays_neg INTEGER,
    mean_epa_per_play DOUBLE, std_epa_per_play DOUBLE, percentile_ev DOUBLE,
    uncertainty_ci_low_per60 DOUBLE, uncertainty_ci_high_per60 DOUBLE,
    pm_direct_per60_ev DOUBLE, pm_pooled_per60_ev DOUBLE, shot_quality_per60_ev DOUBLE,
    qualified_ev BOOLEAN, boot_se_per60 DOUBLE,
    PRIMARY KEY (player_id, season)
);
