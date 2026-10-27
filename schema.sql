-- ====================================================================
-- SportsAnalytics Pro: Database Schema Specification
-- Candidate: Ugwumadu Divine | University of Port Harcourt
-- Database: MySQL 8.0 InnoDB
-- ====================================================================

CREATE DATABASE IF NOT EXISTS sportsanalyticspro_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sportsanalyticspro_db;

-- 1. Users Table (Authentication)
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('analyst', 'admin', 'supervisor') DEFAULT 'analyst',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Matches Table
CREATE TABLE IF NOT EXISTS matches (
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    venue VARCHAR(255) NOT NULL,
    team_a_name VARCHAR(100) NOT NULL,
    team_b_name VARCHAR(100) NOT NULL,
    video_filename VARCHAR(255) NOT NULL,
    processing_status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Team Assignments Table (K-Means Centroids)
CREATE TABLE IF NOT EXISTS team_assignments (
    assignment_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT NOT NULL,
    cluster_label TINYINT NOT NULL,
    team_name VARCHAR(100) NOT NULL,
    jersey_centroid JSON NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. Player Tracks Table (Homography Reprojected 105m x 68m Coordinates)
CREATE TABLE IF NOT EXISTS player_tracks (
    track_id INT NOT NULL,
    match_id INT NOT NULL,
    team_id TINYINT NOT NULL,
    frame_number INT NOT NULL,
    x_coordinate FLOAT NOT NULL,
    y_coordinate FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,
    PRIMARY KEY (match_id, track_id, frame_number),
    KEY idx_match_frame (match_id, frame_number),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Player Stats Table (Physical Kinematics & Speeds)
CREATE TABLE IF NOT EXISTS player_stats (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT NOT NULL,
    track_id INT NOT NULL,
    total_distance FLOAT NOT NULL,
    max_speed FLOAT NOT NULL,
    avg_speed FLOAT NOT NULL,
    sprint_count INT NOT NULL,
    dominant_activity VARCHAR(50) NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    UNIQUE KEY uq_match_track (match_id, track_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. Heatmaps Table (Pitch Zone Densities)
CREATE TABLE IF NOT EXISTS heatmaps (
    heatmap_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT NOT NULL,
    track_id INT NOT NULL,
    pitch_zone VARCHAR(50) NOT NULL,
    visit_count INT NOT NULL,
    intensity_value FLOAT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7. Algorithm Benchmarks Table (Dual-Paradigm Comparison for Chapter 4)
CREATE TABLE IF NOT EXISTS algorithm_benchmarks (
    benchmark_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT NOT NULL,
    algorithm_name VARCHAR(50) NOT NULL,
    paradigm ENUM('Traditional CV', 'Modern Deep Learning') NOT NULL,
    fps FLOAT NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    map_50 FLOAT NOT NULL,
    mota FLOAT NOT NULL,
    motp FLOAT NOT NULL,
    id_switches INT NOT NULL,
    processing_time_sec FLOAT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 8. Interactive Annotations Table (Human-in-the-Loop Validation)
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT NOT NULL,
    frame_number INT NOT NULL,
    annotator_user_id INT DEFAULT NULL,
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_w FLOAT NOT NULL,
    bbox_h FLOAT NOT NULL,
    object_class VARCHAR(50) NOT NULL,
    correction_type ENUM('correct', 'missing_box', 'misclassification', 'false_positive') DEFAULT 'correct',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 9. Surveillance Events Table
CREATE TABLE IF NOT EXISTS surveillance_events (
    event_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT NOT NULL,
    event_type ENUM('intrusion', 'line_crossing', 'loitering', 'stationary_object') NOT NULL,
    timestamp_frame INT NOT NULL,
    object_id INT NOT NULL,
    zone_name VARCHAR(100) NOT NULL,
    alert_level ENUM('low', 'medium', 'high') DEFAULT 'medium',
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
