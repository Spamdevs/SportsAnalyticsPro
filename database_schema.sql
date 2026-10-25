
-- SportsAnalytics Pro Database Schema
CREATE TABLE match (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    team_a VARCHAR(50),
    team_b VARCHAR(50),
    status VARCHAR(20) DEFAULT 'Completed'
);

CREATE TABLE player_stat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    track_id INTEGER,
    team VARCHAR(10),
    distance FLOAT,
    max_speed FLOAT,
    activity VARCHAR(20),
    FOREIGN KEY (match_id) REFERENCES match (id)
);
