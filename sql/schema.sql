CREATE DATABASE IF NOT EXISTS credit_risk_system;

USE credit_risk_system;

CREATE TABLE IF NOT EXISTS customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    date_of_birth DATE,
    gender ENUM('Male', 'Female', 'Other') NULL,
    marital_status ENUM('Single', 'Married', 'Divorced', 'Widowed') NULL,
    employment_status ENUM('Employed', 'Business', 'Retired', 'Student', 'Unemployed') NULL,
    annual_income DECIMAL(10, 2),
    city VARCHAR(100),
    state VARCHAR(100),
    account_created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS loan (
    loan_id INT AUTO_INCREMENT PRIMARY KEY,
    loan_name VARCHAR(100),
    customer_id INT,
    loan_amount DECIMAL(20, 2),
    intrest_rate DECIMAL(5, 2),
    loan_terms_month INT,
    disdursment_date DATE,
    loan_status ENUM('Active', 'Close', 'Default') DEFAULT 'Active',
    loan_type ENUM('Home', 'Education', 'Business', 'Personal', 'Gold', 'Agriculture'),
    colletral_type VARCHAR(150),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS repayments (
    repayment_id INT AUTO_INCREMENT PRIMARY KEY,
    loan_id INT NOT NULL,
    due_date DATE NOT NULL,
    payment_date DATE NULL,
    emi_amount DECIMAL(15, 2),
    principle_component DECIMAL(15, 2) NULL,
    intrest_component DECIMAL(15, 2) NULL,
    days_late INT DEFAULT 0,
    status ENUM('Paid', 'Missed', 'Partial') DEFAULT 'Missed',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (loan_id) REFERENCES loan(loan_id)
);

CREATE OR REPLACE VIEW vm_model_dataset AS
SELECT
    l.loan_id,
    l.loan_name,
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS full_name,
    c.date_of_birth,
    c.gender,
    c.marital_status,
    c.employment_status,
    c.annual_income,
    c.city,
    c.state,
    l.loan_amount,
    l.intrest_rate,
    l.loan_terms_month,
    l.disdursment_date,
    l.loan_status,
    l.loan_type,
    l.colletral_type,
    COALESCE(AVG(r.days_late), 0) AS avg_days_late,
    COALESCE(SUM(CASE WHEN r.days_late > 30 THEN 1 ELSE 0 END), 0) AS num_30_plus_late,
    COALESCE(SUM(CASE WHEN r.status = 'Missed' THEN 1 ELSE 0 END), 0) AS num_missed_payments,
    COALESCE(COUNT(r.repayment_id), 0) AS total_sheduled_payments,
    CASE WHEN l.loan_status = 'Default' THEN 1 ELSE 0 END AS default_flag
FROM loan AS l
JOIN customers AS c
    ON l.customer_id = c.customer_id
LEFT JOIN repayments AS r
    ON l.loan_id = r.loan_id
GROUP BY
    l.loan_id,
    l.loan_name,
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name),
    c.date_of_birth,
    c.gender,
    c.marital_status,
    c.employment_status,
    c.annual_income,
    c.city,
    c.state,
    l.loan_amount,
    l.intrest_rate,
    l.loan_terms_month,
    l.disdursment_date,
    l.loan_status,
    l.loan_type,
    l.colletral_type;

CREATE TABLE IF NOT EXISTS model_score (
    score_id INT AUTO_INCREMENT PRIMARY KEY,
    loan_id INT NOT NULL,
    customer_id INT NOT NULL,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    prob_default DECIMAL(6, 4),
    risk_band ENUM('Low', 'Medium', 'High', 'Very High'),
    model_version VARCHAR(100),
    UNIQUE KEY uq_model_score_loan_id (loan_id),
    FOREIGN KEY (loan_id) REFERENCES loan(loan_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS model_score_rejections (
    rejection_id INT AUTO_INCREMENT PRIMARY KEY,
    loan_id INT NOT NULL,
    customer_id INT NULL,
    rejection_stage VARCHAR(50) NOT NULL,
    rejection_reasons TEXT NOT NULL,
    source_script VARCHAR(100) NOT NULL,
    snapshot_payload LONGTEXT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_model_score_rejections_loan_id (loan_id)
);
