### 1. Statistical Significance Testing (Medium Priority)
Current state: We compute metrics but don't test if differences are statistically significant
What's needed:
- Diebold-Mariano test (compare forecast accuracy)
- Model Confidence Set (identify best model set)
- Giacomini-White test (conditional predictive ability)
- Per-regime performance comparison
Why it matters: Can't claim "MoE is better" without proving it statistically
Implementation: src/evaluation/statistical_tests.py
---
### 2. Comprehensive Evaluation Script (High Priority)
Current state: Each training script saves its own results, no unified comparison
What's needed:
- Single script that loads ALL model results
- Generates comparison tables (HAR-RV vs LSTM vs TCN vs MoE)
- Runs statistical tests
- Creates publication-quality comparison plots
- Exports LaTeX tables for papers
- Per-regime breakdown (which model wins in high vol vs low vol?)
Why it matters: This is what you'll use for your final report/presentation
Implementation: scripts/evaluate_all.py
---
### 3. Gating Weight Analysis (High Priority)
Current state: We train MoE but don't analyze what it learned
What's needed:
- Extract gating weights from trained MoE models
- Analyze which expert is used when
- Correlation: gating weights vs market conditions
- Verify: Does gating use HAR-RV in calm markets, LSTM in volatile markets?
- Case studies: COVID crash, 2008 crisis - which expert was activated?
Why it matters: Core research question - does MoE learn meaningful patterns?
Implementation: scripts/analyze_gating.py
---
### 4. Visualization Suite (Medium Priority)
Current state: Basic plotting functions exist but aren't integrated
What's needed:
- Script to generate all plots automatically
- Forecast comparison plots (actual vs predicted for all models)
- Regime timeline with model performance overlay
- Gating weight heatmaps over time
- Error distribution plots
- Per-regime error analysis
Why it matters: Visualizations are crucial for understanding and presenting results
Implementation: scripts/generate_plots.py
---
### 5. Test Set Evaluation (High Priority)
Current state: We only evaluate on validation set
What's needed:
- Run all models on test set (2022-2025)
- Final performance metrics
- Out-of-sample generalization check
- Compare val vs test performance (overfitting check)
Why it matters: True model performance is measured on unseen test data
Implementation: Update existing scripts or create scripts/test_evaluation.py
---
### 6. Ablation Studies (Medium Priority)
Current state: We train full MoE, but don't know what contributes to performance
What's needed:
- MoE vs best single expert
- MoE with 2 experts vs 3 experts
- Gating with regime supervision vs without
- Impact of freezing experts vs joint fine-tuning
- Different gating architectures (shallow vs deep)
Why it matters: Understand what actually helps and what doesn't
Implementation: scripts/ablation_studies.py
---
### 7. Error Analysis (Low Priority but Valuable)
Current state: We know overall RMSE, but not where/when models fail
What's needed:
- Identify worst prediction periods
- Analyze: What happened during high-error periods?
- Correlation: errors vs market events, volatility spikes
- Compare: Do all models fail at same times or different times?
Why it matters: Understand model limitations and failure modes
Implementation: notebooks/error_analysis.ipynb or script
---
### 8. Configuration Validation (Low Priority)
Current state: Config file can have invalid values, no checking
What's needed:
- Validate config on startup
- Check: paths exist, parameters in valid ranges
- Helpful error messages
- Schema validation (using pydantic or similar)
Why it matters: Better user experience, catch errors early
Implementation: src/utils/config_validator.py
---
### 9. Logging (Low Priority)
Current state: Print statements everywhere
What's needed:
- Proper logging with levels (DEBUG, INFO, WARNING, ERROR)
- Log files saved to outputs/logs/
- Configurable verbosity
- Timestamps on all logs
Why it matters: Professional code, easier debugging
Implementation: Add logging throughout codebase
---
### 10. Foundation Model Integration (Optional - Stretch Goal)
Current state: Mentioned in proposal but not implemented
What's needed:
- Chronos model wrapper (Hugging Face)
- Fine-tuning on volatility data
- Add as 4th expert to MoE
- Compare: Does foundation model help?
Why it matters: Original proposal mentioned it, would be cool to include
Implementation: src/models/foundation.py, update MoE
---
## Recommended Priority Order
### Phase 1: Complete Core Functionality (Week 7)
1. Test Set Evaluation - Run all models on test data
2. Comprehensive Evaluation Script - Unified comparison
3. Statistical Significance Testing - Prove MoE is better
4. Gating Weight Analysis - Understand what MoE learned
Deliverable: Complete results with statistical validation
### Phase 2: Presentation & Reporting (Post-Week 7)
5. Visualization Suite - Generate all plots
6. Error Analysis - Understand failures
7. Ablation Studies - What contributes to performance
Deliverable: Publication-ready figures and tables
### Phase 3: Polish (If Time Permits)
8. Configuration Validation - Better UX
9. Logging - Professional code
10. Foundation Models - Stretch goal