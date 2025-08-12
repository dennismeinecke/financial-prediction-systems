# Changelog

All notable changes to the SP 500 and Ethereum prediction systems will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-01-08

### Added
- Risk management framework with Value at Risk (VaR) calculations
- Additional technical indicators (ATR, CCI, Williams %R, ADX, OBV, ROC)
- Cross-market analysis features (BTC correlation for Ethereum system)
- Market regime detection and classification
- Risk-adjusted prediction algorithms
- Comprehensive risk metrics reporting
- VIX integration for market sentiment analysis
- Enhanced model evaluation with proper sample alignment
- Data validation and cleaning procedures
- Prediction clipping to prevent unrealistic outputs
- Confidence interval calculations
- Detailed audit logging capabilities

### Changed
- Optimized lookback period from 5y to 2y for more relevant recent data
- Reduced sequence length from 30 to 20 for better data availability
- Simplified model architectures for faster training
- Improved error handling and exception management
- Enhanced data cleaning and infinity value handling
- Updated model evaluation methodology
- Modified prediction ranges for crypto volatility (±20% for ETH vs ±10% for SPX)

### Fixed
- Sample count mismatch errors in model evaluation
- Infinity and NaN value propagation in calculations
- TensorFlow and protobuf warning messages
- Model loading and saving procedures
- Prediction clipping edge cases
- Data alignment issues between features and targets

### Security
- Added input validation for all data sources
- Implemented proper error handling for external API calls
- Added data sanitization procedures
- Enhanced model persistence security

## [1.0.0] - 2023-12-15

### Added
- Initial release of SP 500 prediction system
- Initial release of Ethereum prediction system
- Multi-model ensemble approach (Random Forest, LSTM, GRU, Transformer)
- Technical indicator calculations (RSI, MACD, Bollinger Bands, etc.)
- Model training and evaluation framework
- Prediction reporting system
- Model persistence and loading capabilities