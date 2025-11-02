# Visualization System Implementation Details

**Date**: 2025-10-28
**Files Analyzed**: `src/training/extended_train_script.py`, `src/evaluation/test_trained_model.py`

---

## Data Collection Architecture

### Training Metrics Collection

**ModelTracker Callback** (lines 210-308, extended_train_script.py)
- **Metrics tracked**:
  - `timesteps`: List[int] - cumulative environment steps
  - `episode_rewards`: List[float] - total reward per episode
  - `episode_lengths`: List[int] - steps per episode
  - `reward_components`: Dict[str, List[float]] - decomposed reward signals
- **Collection timing**: Every `n_steps` calls to `on_step()` method
- **Storage**: In-memory lists, saved to JSON on training completion
- **Trigger**: `locals['dones'][0] == True` (episode termination)

**EnhancedMetricsCallback** (lines 311-436, extended_train_script.py)
- **Additional metrics**:
  - `goals_scored`: int - cumulative goal count
  - `ball_possession_time`: float - timesteps with ball distance < 0.3m
  - `collisions`: int - robot-ball collision count
  - `out_of_bounds`: int - ball boundary violation count
  - `goal_approaches`: int - ball enters attacking third (>0.66 × field_width)
- **Collection method**: Access `locals['infos'][0]` dict from environment
- **Aggregation**: Running totals updated per episode
- **Output files**: `enhanced_metrics.json`, `training_summary.txt`

### Evaluation Metrics Collection

**evaluate_model_comprehensive()** (lines 554-687, extended_train_script.py)
- **Per-episode metrics**:
  - `episode_rewards`: List[float]
  - `episode_lengths`: List[int]
  - `goals_scored_list`: List[int] (0 or 1)
  - `ball_possession_timesteps_list`: List[int]
  - `robot_collisions_count_list`: List[int] (binary)
  - `ball_out_of_bounds_count_list`: List[int] (binary)
  - `final_ball_distance_list`: List[float] (meters)
  - `goal_approaches_list`: List[int]
- **Thresholds**:
  - Possession: 0.3m (converted to pixels via `field_config.METERS_TO_PIXELS`)
  - Collision: `unwrapped_env.collision_distance`
  - Attacking third: `field_width * 0.66`
- **Return structure**:
```python
{
    'mean_reward': float,
    'std_reward': float,
    'episode_rewards': List[float],
    'episode_lengths': List[int],
    'goals_scored': int,
    'enhanced_metrics': {
        'ball_possession_rate': float,
        'collision_rate': float,
        'out_of_bounds_rate': float,
        # ... additional rates
    }
}
```

**compare_three_policies()** (lines 932-1232, extended_train_script.py)
- **Evaluation loop**: 50 episodes per policy
- **Data collection**: Identical to `evaluate_model_comprehensive()`
- **Special handling**: OpponentAsPolicy requires `env` reference via `policy.set_env(env)`
- **Action smoothing**: ActionSmoothingWrapper with factor=0.6
- **Outputs**:
  1. Plots via `create_3way_comparison_plots()`
  2. Summary text via `generate_thesis_summary()`
  3. JSON via `save_3way_results_json()`
  4. Markdown via `save_3way_results_markdown()`

### Storage Formats

**JSON Structure** (save_3way_results_json, lines 1057-1123)
```json
{
    "timestamp": "YYYYMMDD_HHMMSS",
    "metadata": {
        "ddpg_model_path": str,
        "ppo_model_path": str,
        "episodes_per_policy": int,
        "difficulty": str
    },
    "results": {
        "ddpg": {
            "mean_reward": float,
            "std_reward": float,
            "total_goals": int,
            "ball_possession_rate": float,
            "collision_rate": float,
            "mean_episode_length": float,
            "out_of_bounds_rate": float,
            "episode_rewards": List[float],
            "episode_lengths": List[int]
        },
        "ppo": { ... },
        "handcoded": { ... }  // Optional
    }
}
```

**Markdown Tables** (save_3way_results_markdown, lines 1126-1232)
- Format: GitHub-flavored markdown
- Tables: Summary statistics, detailed metrics, episode-by-episode breakdown
- Precision: 2 decimal places for floats, percentages for rates

---

## Visualization Functions

### 1. create_academic_training_curves()

**Location**: Lines 2094-2313, extended_train_script.py

**Purpose**: Generate publication-quality 4-panel training analysis plot

**Data Sources**:
- `training_data` dict with keys:
  - `'timesteps'`: List[int]
  - `'rewards'`: List[float]
  - `'episode_lengths'`: List[int]
  - `'reward_components'`: Dict[str, List[float]] (optional)

**Processing**:
- **Smoothing**: Rolling window average
  - Window size: `max(len(rewards) // 50, 10)`
  - Method: `pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()`
  - Standard deviation: Same window for confidence intervals
- **Adaptive scaling**:
  - If `max(timesteps) < 1_000_000`: divide by 1000, label "K steps"
  - Else: divide by 1_000_000, label "M steps"
  - Code: `timesteps_scaled = [t / scale_factor for t in timesteps]`

**Plot Configuration**:
- Figure size: `(18, 14)`
- DPI: `300`
- Style: `plt.style.use('seaborn-v0_8-paper')`
- Layout: `fig, axes = plt.subplots(2, 2)`
- Spacing: `plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.88)`
- Font sizes:
  - Labels: 14
  - Titles: 16
  - Suptitle: 18

**Panels**:
1. **Learning Curve** (ax[0,0]):
   - Scatter: Raw rewards (alpha=0.3, s=20)
   - Line: Smoothed rewards (linewidth=2.5)
   - Fill: ±1 std confidence interval (alpha=0.2)
   - Grid: alpha=0.3

2. **Episode Length Distribution** (ax[0,1]):
   - Histogram: bins=30, alpha=0.7, edgecolor='black'
   - Stats: Mean line (red dashed, linewidth=2)
   - Annotation: Mean value displayed

3. **Reward Components Timeline** (ax[1,0]):
   - If `reward_components` provided: Stacked area plot
   - Else: Episode length progression (scatter + smoothed line)

4. **Performance Statistics** (ax[1,1]):
   - Text summary with metrics:
     - Final mean reward (last 10%)
     - Best reward
     - Convergence indicator
     - Training duration
   - Format: Centered text, fontsize=12

**Output**:
- Filename: `{algorithm}_training_analysis_{timestamp}.png`
- Format: PNG
- DPI: 300

---

### 2. create_algorithm_comparison_plot()

**Location**: Lines 2314-2644, extended_train_script.py

**Purpose**: Compare PPO vs DDPG with 6-panel analysis and statistical validation

**Data Sources**:
- `ppo_data` dict: Same structure as `training_data` above
- `ddpg_data` dict: Same structure as `training_data` above

**Processing**:
- **Smoothing**: Same rolling window method as `create_academic_training_curves()`
- **Adaptive scaling**: Same K/M logic
- **Statistical tests**:
  - Independent t-test: `scipy.stats.ttest_ind(ppo_rewards_final, ddpg_rewards_final)`
  - Cohen's d: `(mean_ppo - mean_ddpg) / pooled_std`
  - Effect size classification: small (0.2), medium (0.5), large (0.8)

**Plot Configuration**:
- Figure size: `(24, 14)`
- Layout: `fig, axes = plt.subplots(2, 3)`
- Colors:
  - PPO: `'#2E86C1'` (blue)
  - DDPG: `'#E74C3C'` (red)
- Spacing: `hspace=0.3, wspace=0.25, top=0.93`

**Panels**:
1. **Training Curves Comparison** (ax[0,0]):
   - PPO: Line + confidence band
   - DDPG: Line + confidence band
   - Legend: loc='best', fontsize=12

2. **Episode Length Comparison** (ax[0,1]):
   - PPO: Smoothed line
   - DDPG: Smoothed line
   - Y-axis: Episode length (steps)

3. **Reward Distribution** (ax[0,2]):
   - Histograms: Last 25% of training episodes
   - Bins: 30
   - Alpha: 0.6
   - Overlay: Both distributions

4. **Sample Efficiency** (ax[1,0]):
   - X-axis: Timesteps (scaled)
   - Y-axis: Cumulative mean reward
   - Lines: PPO vs DDPG

5. **Statistical Summary** (ax[1,1]):
   - Text display:
     - Mean rewards ± std
     - t-statistic
     - p-value
     - Cohen's d
     - Effect size label
     - Significance indicator
   - Format: Left-aligned, fontsize=11

6. **Enhanced Metrics Radar** (ax[1,2]):
   - If `enhanced_metrics` available:
     - Radar plot with 5-6 axes
     - Metrics: possession, goals, collisions, etc.
     - Normalized to [0, 1]
   - Else: "Not available" text

**Output**:
- Filename: `algorithm_comparison_{timestamp}.png`
- Format: PNG
- DPI: 300

---

### 3. create_enhanced_metrics_comparison()

**Location**: Lines 2644-2843, extended_train_script.py

**Purpose**: Visualize soccer-specific performance metrics across difficulty levels

**Data Sources**:
- `metrics_data` dict with keys:
  - `'easy'`, `'medium'`, `'hard'` (difficulty levels)
  - Each contains: `goals`, `possession`, `collisions`, `episode_lengths`, `rewards`

**Plot Configuration**:
- Figure size: `(20, 12)`
- Layout: `fig, axes = plt.subplots(2, 3)`
- Colors: Distinct color per difficulty (e.g., green, orange, red)

**Panels**:
1. **Goals Scored**: Bar chart across difficulties
2. **Ball Possession Rate**: Bar chart with percentage labels
3. **Collision Rate**: Bar chart (lower is better)
4. **Mean Episode Length**: Bar chart with error bars
5. **Mean Reward**: Bar chart with error bars
6. **Success Rate Heatmap**: Grid showing pass/fail per difficulty

**Output**:
- Filename: `enhanced_metrics_comparison_{timestamp}.png`
- Format: PNG
- DPI: 300

---

### 4. create_3way_comparison_plots()

**Location**: Lines 1235-1407, extended_train_script.py

**Purpose**: Generate bar charts, histograms, and line plots for DDPG vs PPO vs Hand-Coded

**Data Sources**:
- `results` dict with keys: `'ddpg'`, `'ppo'`, `'handcoded'` (optional)
- Each contains: `mean_reward`, `std_reward`, `total_goals`, `ball_possession_rate`, etc.

**Plot Configuration**:
- **Figure 1 (Bar Charts)**:
  - Size: `(18, 10)`
  - Layout: `2 rows × 3 cols` (6 metrics)
  - Colors: DDPG='#2E86C1', PPO='#E74C3C', Hand-coded='#27AE60'

- **Figure 2 (Histograms)**:
  - Size: `(6 * num_policies, 5)`
  - Layout: `1 row × num_policies cols`
  - Bins: 20
  - Alpha: 0.7

- **Figure 3 (Line Plot)**:
  - Size: `(12, 6)`
  - X-axis: Episode number
  - Y-axis: Cumulative goals

**Metrics Visualized**:
1. Mean Reward (± std error bars)
2. Total Goals Scored
3. Ball Possession Rate (%)
4. Collision Rate (%)
5. Mean Episode Length (± std)
6. Out of Bounds Rate (%)

**Outputs**:
1. `2way_comparison_bar_charts.png` (300 DPI)
2. `ddpg_vs_ppo_reward_histograms.png` (300 DPI)
3. `2way_goals_over_time.png` (300 DPI)

---

### 5. generate_thesis_summary()

**Location**: Lines 1410-1553, extended_train_script.py

**Purpose**: Generate markdown-formatted text summary for thesis inclusion

**Data Sources**:
- `results` dict (same as `create_3way_comparison_plots()`)

**Processing**:
- String concatenation with f-string interpolation
- Statistical formatting: `.2f` for floats, `.1f` for percentages
- Conditional sections: Includes hand-coded only if present

**Output Structure**:
```markdown
# DDPG vs PPO Policy Comparison

## Comparative Analysis
### Mean Episode Reward
- DDPG: {value} ± {std}
- PPO: {value} ± {std}

### Total Goals Scored
...

### Ball Possession
...

### Collision Rate
...

### Episode Length
...

### Out of Bounds Rate
...

## Key Findings
- Winner: {policy_name}
- Performance margin: {percentage}%
- Statistical significance: {if diff > 2*std}
```

**Output**:
- Filename: `2way_comparison_summary.txt`
- Format: Plain text (markdown-formatted)

---

### 6. plot_game_performance() [test_trained_model.py]

**Location**: Lines 450-580, test_trained_model.py

**Purpose**: Generate 4-panel game performance visualization for single model evaluation

**Data Sources**:
- `results` dict with keys:
  - `'episode_rewards'`: List[float]
  - `'episode_lengths'`: List[int]
  - `'goals_scored'`: int
  - `'possession_rate'`: float
  - `'collision_rate'`: float
  - Additional metrics from evaluation

**Plot Configuration**:
- Figure size: `(16, 12)`
- Layout: `2 × 2`
- DPI: `300`
- Style: `'seaborn-v0_8-paper'`

**Panels**:
1. **Rewards Over Episodes**: Line plot + scatter
2. **Episode Length Distribution**: Histogram
3. **Metrics Summary**: Text display with key statistics
4. **Success Metrics**: Horizontal bar chart (goals, possession, etc.)

**Output**:
- Filename: `game_performance_{model_type}_{difficulty}_{timestamp}.png`
- Format: PNG
- DPI: 300

---

### 7. plot_learning_analysis() [test_trained_model.py]

**Location**: Lines 583-710, test_trained_model.py

**Purpose**: Analyze learning progression and stability for single model

**Data Sources**:
- `results` dict (same as `plot_game_performance()`)

**Processing**:
- **Temporal analysis**: Split episodes into quartiles (25%, 50%, 75%, 100%)
- **Trend detection**: Linear regression on rewards over time
- **Stability metric**: Coefficient of variation (std / mean)

**Plot Configuration**:
- Figure size: `(16, 10)`
- Layout: `2 × 2`
- DPI: `300`

**Panels**:
1. **Learning Progression**: Rewards by quartile (box plot)
2. **Stability Analysis**: Rolling mean + std bands
3. **Trend Line**: Linear fit with equation annotation
4. **Quartile Statistics**: Text summary table

**Output**:
- Filename: `learning_analysis_{model_type}_{difficulty}_{timestamp}.png`
- Format: PNG
- DPI: 300

---

### 8. plot_comparison() [test_trained_model.py]

**Location**: Lines 713-890, test_trained_model.py

**Purpose**: Compare two models side-by-side with statistical tests

**Data Sources**:
- `results1` dict: First model metrics
- `results2` dict: Second model metrics

**Processing**:
- **Statistical tests**:
  - Independent t-test: `scipy.stats.ttest_ind()`
  - Effect size: Cohen's d
  - Confidence intervals: 95% via `scipy.stats.t.interval()`

**Plot Configuration**:
- Figure size: `(20, 12)`
- Layout: `2 × 3`
- DPI: `300`
- Colors: Model 1='#3498db', Model 2='#e74c3c'

**Panels**:
1. **Reward Comparison**: Side-by-side box plots
2. **Episode Length Comparison**: Side-by-side box plots
3. **Win Rate Comparison**: Bar chart (goals scored)
4. **Possession vs Collisions**: Scatter plot (2D comparison)
5. **Statistical Summary**: Text with t-test results
6. **Reward Distributions**: Overlapping histograms

**Output**:
- Filename: `comparison_{model1}_vs_{model2}_{timestamp}.png`
- Format: PNG
- DPI: 300
- Includes statistical annotations: p-values, effect sizes, significance stars

---

## Key Implementation Details

### Plotting Standards

**Global Settings**:
- DPI: `300` (all saved figures)
- Style: `'seaborn-v0_8-paper'`
- Font family: Default (usually DejaVu Sans)
- Font sizes:
  - Body text: 12
  - Axis labels: 14
  - Subplot titles: 16
  - Figure suptitles: 18

**Color Palette**:
- PPO: `'#2E86C1'` (medium blue)
- DDPG: `'#E74C3C'` (red-orange)
- Hand-coded: `'#27AE60'` (green)
- Confidence bands: Same as line color, alpha=0.2
- Grid lines: alpha=0.3

**Figure Sizing**:
- Training analysis (4-panel): `(18, 14)`
- Algorithm comparison (6-panel): `(24, 14)`
- Enhanced metrics: `(20, 12)`
- 3-way bar charts: `(18, 10)`
- Single model evaluation: `(16, 12)`

### Data Aggregation Methods

**Smoothing**:
- Method: Rolling window average via pandas
- Window size: `max(len(data) // 50, 10)` (adaptive, minimum 10)
- Implementation:
  ```python
  smoothed = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
  ```
- Applied to: Episode rewards, episode lengths, timesteps

**Statistical Calculations**:
- Mean: `np.mean(data)`
- Standard deviation: `np.std(data, ddof=1)` (sample std)
- Standard error: `std / np.sqrt(len(data))`
- Confidence interval: `mean ± 1.96 * std_error` (95%)

**Rate Calculations**:
- Ball possession rate: `sum(possession_timesteps) / sum(episode_lengths)`
- Collision rate: `sum(collision_counts) / num_episodes`
- Out of bounds rate: `sum(oob_counts) / num_episodes`
- Goal approach rate: `sum(goal_approaches) / num_episodes`

### Callback System

**ModelTracker Callback**:
- Inherits: `BaseCallback` (stable_baselines3)
- Trigger: `on_step()` method called every environment step
- Storage: Class attributes (lists)
- Thread safety: Not required (single-threaded training)

**EnhancedMetricsCallback**:
- Inherits: `BaseCallback`
- Access method: `self.locals['infos'][0]` for environment data
- Logging: Optional TensorBoard integration
- File output: JSON + text summary on training completion

**Integration**:
```python
callbacks = [
    ModelTracker(),
    EnhancedMetricsCallback(log_freq=1000)
]
model.learn(total_timesteps=2_500_000, callback=callbacks)
```

### File Organization

**Directory Structure**:
```
{output_dir}/
├── models/
│   ├── {algorithm}/
│   │   ├── best_model_{timestamp}.zip
│   │   ├── checkpoint_{step}_model.zip
│   │   └── final_model.zip
├── plots/
│   ├── {algorithm}_training_analysis_{timestamp}.png
│   ├── algorithm_comparison_{timestamp}.png
│   ├── 2way_comparison_bar_charts.png
│   └── 2way_goals_over_time.png
├── data/
│   ├── training_data_{algorithm}.json
│   ├── enhanced_metrics.json
│   └── 2way_comparison_results.json
└── summaries/
    ├── training_summary.txt
    └── 2way_comparison_summary.txt
```

**Naming Convention**:
- Timestamps: `YYYYMMDD_HHMMSS` format
- Algorithm names: Lowercase (ppo, ddpg)
- Difficulty levels: Lowercase (easy, medium, hard)
- File prefixes: Descriptive (e.g., `2way_`, `enhanced_`, `comparison_`)

### Export Formats

**PNG Images**:
- Resolution: 300 DPI
- Color mode: RGB
- Compression: Default matplotlib PNG compression
- Metadata: Timestamp in filename

**JSON Files**:
- Indentation: 2 spaces
- Float precision: Full precision (no rounding in JSON)
- Arrays: Nested lists for episode data
- Encoding: UTF-8

**Text Summaries**:
- Format: Markdown-compatible plain text
- Line endings: Unix (`\n`)
- Float formatting: `.2f` for most values, `.1f` for percentages
- Encoding: UTF-8

---

**End of Implementation Details**
