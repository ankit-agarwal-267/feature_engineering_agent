# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-04-22

### Added
- **Core Engine:** Support for both Pandas and Polars backends.
- **Semantic Profiler:** Automatic detection of 12+ semantic types with overrides.
- **Interaction Engine:** Numeric/Categorical interactions and group statistics with HITL approval.
- **Leakage Guard:** Automated detection of target leakage based on correlation and heuristics.
- **Decision Log & Audit Reporter:** Machine-readable logs and human-readable Markdown reports.
- **Data Ingestion:** Support for CSV, Parquet, JSON, and SQL.
- **Documentation:** User Manual, Quickstart, and Configuration Reference.

### Fixed
- Issue where NaNs caused errors during integer casting in numeric transforms.
- Inconsistent datetime parsing for non-standard formats.
- Correct identification of high-cardinality categorical columns.

### Changed
- Improved error handling for missing dependencies like `scipy` or `scikit-learn`.
- Standardized column naming convention for derived features.
