-- ********* TODO Add more comments **********


-- Codex DS Showcase: SQL Analysis Queries
-- ==========================================
-- This file demonstrates SQL fluency for analyzing developer telemetry data.
-- These queries mirror the type of analysis a Codex Data Scientist would run
-- on production telemetry databases.

-- Note: These queries are designed for SQLite/PostgreSQL syntax.
-- In production, telemetry would typically be stored in a data warehouse
-- (BigQuery, Snowflake, Redshift, etc.)

-- ==========================================
-- 1. BASIC METRICS
-- ==========================================

-- Overall acceptance rate
SELECT 
    COUNT(*) as total_suggestions,
    SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) as accepted_count,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as acceptance_rate
FROM telemetry_events;

-- Acceptance rate by model version (A/B comparison)
SELECT 
    model_version,
    COUNT(*) as total_suggestions,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as acceptance_rate,
    AVG(latency_ms) as avg_latency_ms,
    AVG(CASE WHEN compile_success = 1 THEN 1.0 ELSE 0.0 END) as compile_success_rate,
    AVG(CASE WHEN test_pass = 1 THEN 1.0 ELSE 0.0 END) as test_pass_rate
FROM telemetry_events
GROUP BY model_version
ORDER BY model_version;

-- ==========================================
-- 2. SEGMENTATION ANALYSIS
-- ==========================================

-- Acceptance rate by user segment
SELECT 
    user_segment,
    COUNT(*) as total_suggestions,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as acceptance_rate,
    AVG(latency_ms) as avg_latency_ms
FROM telemetry_events
GROUP BY user_segment
ORDER BY acceptance_rate DESC;

-- Acceptance rate by language
SELECT 
    language,
    COUNT(*) as total_suggestions,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as acceptance_rate,
    AVG(CASE WHEN compile_success = 1 THEN 1.0 ELSE 0.0 END) as compile_success_rate
FROM telemetry_events
GROUP BY language
ORDER BY acceptance_rate DESC;

-- Combined segmentation: model version + language
SELECT 
    model_version,
    language,
    COUNT(*) as total_suggestions,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as acceptance_rate
FROM telemetry_events
GROUP BY model_version, language
ORDER BY model_version, acceptance_rate DESC;

-- ==========================================
-- 3. LATENCY ANALYSIS
-- ==========================================

-- Latency percentiles
SELECT 
    model_version,
    COUNT(*) as count,
    AVG(latency_ms) as avg_latency,
    MIN(latency_ms) as min_latency,
    MAX(latency_ms) as max_latency,
    -- Approximate percentiles (SQLite doesn't have PERCENTILE_CONT)
    (SELECT latency_ms FROM telemetry_events e2 
     WHERE e2.model_version = e1.model_version 
     ORDER BY latency_ms LIMIT 1 OFFSET (SELECT COUNT(*) FROM telemetry_events e3 WHERE e3.model_version = e1.model_version) * 50 / 100) as p50_latency,
    (SELECT latency_ms FROM telemetry_events e2 
     WHERE e2.model_version = e1.model_version 
     ORDER BY latency_ms LIMIT 1 OFFSET (SELECT COUNT(*) FROM telemetry_events e3 WHERE e3.model_version = e1.model_version) * 95 / 100) as p95_latency
FROM telemetry_events e1
GROUP BY model_version;

-- Latency impact on acceptance (binned)
SELECT 
    CASE 
        WHEN latency_ms < 200 THEN '< 200ms'
        WHEN latency_ms < 500 THEN '200-500ms'
        WHEN latency_ms < 1000 THEN '500-1000ms'
        WHEN latency_ms < 2000 THEN '1000-2000ms'
        ELSE '> 2000ms'
    END as latency_bucket,
    COUNT(*) as total_suggestions,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as acceptance_rate
FROM telemetry_events
GROUP BY latency_bucket
ORDER BY 
    CASE latency_bucket
        WHEN '< 200ms' THEN 1
        WHEN '200-500ms' THEN 2
        WHEN '500-1000ms' THEN 3
        WHEN '1000-2000ms' THEN 4
        ELSE 5
    END;

-- ==========================================
-- 4. SESSION-LEVEL PRODUCTIVITY METRICS
-- ==========================================

-- Session productivity metrics
SELECT 
    session_id,
    developer_id,
    user_segment,
    model_version,
    COUNT(*) as suggestions_in_session,
    SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) as accepted_count,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as session_acceptance_rate,
    AVG(latency_ms) as avg_session_latency,
    SUM(CASE WHEN compile_success = 1 THEN 1 ELSE 0 END) as compile_success_count,
    SUM(CASE WHEN test_pass = 1 THEN 1 ELSE 0 END) as test_pass_count,
    -- Task completion: sessions where at least one suggestion led to test pass
    MAX(CASE WHEN test_pass = 1 THEN 1 ELSE 0 END) as task_completed
FROM telemetry_events
GROUP BY session_id, developer_id, user_segment, model_version
ORDER BY session_acceptance_rate DESC
LIMIT 20;

-- Average session productivity by user segment
SELECT 
    user_segment,
    COUNT(DISTINCT session_id) as total_sessions,
    AVG(session_suggestions) as avg_suggestions_per_session,
    AVG(session_acceptance_rate) as avg_session_acceptance_rate,
    AVG(task_completed) as task_completion_rate
FROM (
    SELECT 
        session_id,
        user_segment,
        COUNT(*) as session_suggestions,
        AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as session_acceptance_rate,
        MAX(CASE WHEN test_pass = 1 THEN 1 ELSE 0 END) as task_completed
    FROM telemetry_events
    GROUP BY session_id, user_segment
) session_metrics
GROUP BY user_segment;

-- ==========================================
-- 5. ERROR ANALYSIS
-- ==========================================

-- Error type distribution
SELECT 
    error_type,
    COUNT(*) as error_count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telemetry_events) as error_percentage
FROM telemetry_events
WHERE error_type != 'none'
GROUP BY error_type
ORDER BY error_count DESC;

-- Error rate by model version
SELECT 
    model_version,
    error_type,
    COUNT(*) as count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telemetry_events e2 WHERE e2.model_version = e1.model_version) as percentage
FROM telemetry_events e1
WHERE error_type != 'none'
GROUP BY model_version, error_type
ORDER BY model_version, count DESC;

-- Hallucination analysis
SELECT 
    model_version,
    COUNT(*) as total_suggestions,
    SUM(CASE WHEN hallucination_flag = 1 THEN 1 ELSE 0 END) as hallucination_count,
    AVG(CASE WHEN hallucination_flag = 1 THEN 1.0 ELSE 0.0 END) as hallucination_rate,
    AVG(CASE WHEN hallucination_flag = 1 AND accepted = 1 THEN 1.0 ELSE 0.0 END) as hallucination_acceptance_rate
FROM telemetry_events
GROUP BY model_version;

-- ==========================================
-- 6. TIME-BASED ANALYSIS
-- ==========================================

-- Acceptance rate over time (daily)
SELECT 
    DATE(timestamp) as date,
    model_version,
    COUNT(*) as suggestions,
    AVG(CASE WHEN accepted = 1 THEN 1.0 ELSE 0.0 END) as acceptance_rate
FROM telemetry_events
GROUP BY DATE(timestamp), model_version
ORDER BY date DESC, model_version;

-- ==========================================
-- 7. TASK-LEVEL ANALYSIS
-- ==========================================

-- Task completion rates
SELECT 
    task_id,
    COUNT(DISTINCT session_id) as sessions_attempted,
    COUNT(*) as total_suggestions,
    SUM(CASE WHEN test_pass = 1 THEN 1 ELSE 0 END) as successful_suggestions,
    AVG(CASE WHEN test_pass = 1 THEN 1.0 ELSE 0.0 END) as task_success_rate
FROM telemetry_events
GROUP BY task_id
ORDER BY task_success_rate DESC;

-- ==========================================
-- 8. ADVANCED: COHORT ANALYSIS
-- ==========================================

-- Developer cohort: first session date
WITH developer_first_session AS (
    SELECT 
        developer_id,
        MIN(DATE(timestamp)) as first_session_date
    FROM telemetry_events
    GROUP BY developer_id
),
session_cohorts AS (
    SELECT 
        e.session_id,
        e.developer_id,
        e.model_version,
        d.first_session_date,
        DATE(e.timestamp) as session_date,
        AVG(CASE WHEN e.accepted = 1 THEN 1.0 ELSE 0.0 END) as session_acceptance_rate
    FROM telemetry_events e
    JOIN developer_first_session d ON e.developer_id = d.developer_id
    GROUP BY e.session_id, e.developer_id, e.model_version, d.first_session_date, DATE(e.timestamp)
)
SELECT 
    first_session_date as cohort,
    model_version,
    COUNT(DISTINCT session_id) as sessions,
    AVG(session_acceptance_rate) as avg_acceptance_rate
FROM session_cohorts
GROUP BY first_session_date, model_version
ORDER BY first_session_date DESC, model_version;

