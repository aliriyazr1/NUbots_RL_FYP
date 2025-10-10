# Statistical Tests Explained - Academic Guide

## Overview

This document explains the statistical tests in `test_statistical_validation.py` for your thesis and research validation.

---

## 1. T-Test (PPO vs DDPG Comparison)

### What It Tests

**Question:** "Is the difference between PPO and DDPG performance **real** or just **random luck**?"

### The Math

```python
t_statistic, p_value = stats.ttest_ind(ppo_rewards, ddpg_rewards)
```

**Inputs:**
- `ppo_rewards`: Array of 20 episode rewards from PPO
- `ddpg_rewards`: Array of 20 episode rewards from DDPG

**Outputs:**
- `t_statistic`: How many standard deviations apart are the means
- `p_value`: Probability this difference is due to random chance

### Hypotheses

- **H₀ (Null Hypothesis)**: PPO mean = DDPG mean (no real difference)
- **H₁ (Alternative Hypothesis)**: PPO mean ≠ DDPG mean (real difference exists)

### Interpreting Results

#### Example 1: Significant Difference

```
PPO Performance:
  Mean:     10,234 ± 892

DDPG Performance:
  Mean:      9,156 ± 1,023

Statistical Comparison:
  t-statistic:    3.456
  p-value:        0.001  ← p < 0.05 ✓
  Significance:   Yes (p < 0.05)
  Cohen's d:      1.12 (large effect)
```

**What This Means:**
- p = 0.001 means there's only a **0.1% chance** this difference is random
- We **reject H₀** (null hypothesis)
- **Conclusion:** "PPO significantly outperformed DDPG, t(38) = 3.456, p = 0.001, d = 1.12"

**For Your Thesis:**
> "Statistical analysis revealed that PPO (M = 10,234, SD = 892) achieved significantly higher rewards than DDPG (M = 9,156, SD = 1,023), t(38) = 3.456, p < 0.001, d = 1.12. This represents a large effect size, indicating PPO's superiority is both statistically significant and practically meaningful."

#### Example 2: No Significant Difference

```
PPO Performance:
  Mean:     10,234 ± 892

DDPG Performance:
  Mean:     10,089 ± 945

Statistical Comparison:
  t-statistic:    0.523
  p-value:        0.603  ← p ≥ 0.05 ✗
  Significance:   No (p >= 0.05)
  Cohen's d:      0.16 (negligible effect)
```

**What This Means:**
- p = 0.603 means there's a **60.3% chance** this difference is just random variation
- We **fail to reject H₀**
- **Conclusion:** "No significant difference found between PPO and DDPG"

**For Your Thesis:**
> "Comparison of PPO (M = 10,234, SD = 892) and DDPG (M = 10,089, SD = 945) revealed no statistically significant difference, t(38) = 0.523, p = 0.603, d = 0.16. Both algorithms demonstrated comparable performance on the soccer task."

---

## 2. P-Value Explained

### What Is It?

**P-value** = Probability of observing this result (or more extreme) if H₀ is true

### Decision Rule

**α = 0.05** (Standard threshold in academic research)

- **p < 0.05** → **Reject H₀** → Result is **statistically significant**
- **p ≥ 0.05** → **Fail to reject H₀** → Result is **NOT significant**

### Common P-values and Interpretation

| P-value | Interpretation | Confidence |
|---------|----------------|------------|
| p < 0.001 | **Highly significant** | Very high confidence |
| p < 0.01 | **Very significant** | High confidence |
| p < 0.05 | **Significant** | Moderate confidence |
| p < 0.10 | **Marginally significant** | Weak confidence |
| p ≥ 0.10 | **Not significant** | No confidence |

### Important: What P-value Does NOT Tell You

❌ **p-value ≠ "Probability H₀ is true"**
❌ **p-value ≠ "Importance of result"**
✅ **p-value = "Probability of data, assuming H₀ is true"**

**Example:**
- p = 0.03 means: "If PPO and DDPG were truly equal, there's only a 3% chance we'd see a difference this large or larger."

---

## 3. Cohen's d (Effect Size)

### What It Tests

**Question:** "How **big** is the difference?" (even if it's statistically significant)

### The Math

```python
cohens_d = (mean_ppo - mean_ddpg) / pooled_std
```

### Interpretation (Cohen, 1988)

| Cohen's d | Interpretation | Meaning |
|-----------|----------------|---------|
| \|d\| < 0.2 | **Negligible** | Almost no practical difference |
| 0.2 ≤ \|d\| < 0.5 | **Small** | Small but noticeable difference |
| 0.5 ≤ \|d\| < 0.8 | **Medium** | Moderate, meaningful difference |
| \|d\| ≥ 0.8 | **Large** | Large, substantial difference |

### Why It Matters

You can have:
1. **Statistically significant but small effect**: p < 0.05, d = 0.15
   - "Yes, there's a difference, but it's tiny"

2. **Large effect but not significant**: p = 0.08, d = 0.90
   - "Big difference, but sample size too small to be sure"

**Best case:** p < 0.05 AND d ≥ 0.5
- Both statistically significant AND practically meaningful

### Example

```
PPO mean: 10,234
DDPG mean: 9,156
Pooled SD: 960

Cohen's d = (10,234 - 9,156) / 960 = 1.12

Interpretation: Large effect size
```

**For Your Thesis:**
> "The effect size was large (d = 1.12), indicating the superiority of PPO represents a substantial and practically meaningful improvement over DDPG."

---

## 4. Confidence Intervals (95% CI)

### What It Shows

**95% Confidence Interval** = "We're 95% confident the **true mean difference** lies within this range"

### Example

```
Mean Difference:
  PPO - DDPG:     1,078
  95% CI:         [432, 1,724]
```

**Interpretation:**
- Best estimate: PPO is 1,078 points better
- We're 95% confident the true difference is between 432 and 1,724
- **Key insight:** The interval does NOT include 0 → Difference is significant!

### Decision Rule

- **CI does NOT include 0** → Significant difference ✓
- **CI includes 0** → No significant difference ✗

**For Your Thesis:**
> "PPO outperformed DDPG by 1,078 points (95% CI [432, 1,724]), confirming a reliable performance advantage."

---

## 5. Complete Example for Your Thesis

### Test Results:

```
PPO Performance:
  Mean:     10,234.12 ± 892.45
  Median:   10,189.00
  Range:    [8,567, 12,103]
  CV:       0.0872

DDPG Performance:
  Mean:      9,156.78 ± 1,023.56
  Median:    9,234.00
  Range:     [7,123, 11,045]
  CV:        0.1118

Statistical Comparison:
  t-statistic:    3.456
  p-value:        0.001
  Significance:   Yes (p < 0.05)
  Cohen's d:      1.12
  Effect size:    Large

Mean Difference:
  PPO - DDPG:     1,077.34
  95% CI:         [432.12, 1,722.56]
```

### How to Report in Thesis:

#### Method Section:
> "Algorithm performance was compared using independent samples t-tests with a significance level of α = 0.05. Effect sizes were calculated using Cohen's d (Cohen, 1988). Each algorithm was evaluated over 20 episodes (n = 20) on the easy difficulty setting."

#### Results Section:
> "PPO (M = 10,234.12, SD = 892.45) significantly outperformed DDPG (M = 9,156.78, SD = 1,023.56), t(38) = 3.456, p < 0.001, 95% CI [432.12, 1,722.56]. The effect size was large (d = 1.12), indicating PPO's advantage represents a substantial and practically meaningful improvement. PPO also demonstrated greater consistency, with a lower coefficient of variation (CV = 0.087) compared to DDPG (CV = 0.112)."

#### Discussion Section:
> "The statistically significant performance difference (p < 0.001) with a large effect size (d = 1.12) provides strong evidence for PPO's superiority on this task. The 95% confidence interval [432.12, 1,722.56] suggests we can be confident the true performance advantage lies between 432 and 1,723 points. This finding aligns with prior research demonstrating PPO's effectiveness in continuous control tasks (Schulman et al., 2017)."

---

## 6. Common Mistakes to Avoid

### ❌ Mistake 1: "p = 0.05 means 5% chance H₀ is true"
✅ **Correct:** "p = 0.05 means 5% chance of observing this data if H₀ were true"

### ❌ Mistake 2: "p < 0.05 means it's important"
✅ **Correct:** Check Cohen's d for practical importance. p-value only shows statistical significance.

### ❌ Mistake 3: "p = 0.049 is significant but p = 0.051 is not"
✅ **Correct:** Don't treat 0.05 as a hard cutoff. Report exact p-values and discuss context.

### ❌ Mistake 4: "No significant difference means they're equal"
✅ **Correct:** "Failure to reject H₀" ≠ "H₀ is true". May need larger sample size.

### ❌ Mistake 5: Ignoring effect size
✅ **Correct:** Always report both p-value AND effect size (Cohen's d)

---

## 7. Why These Tests Matter for Your FYP

### Academic Rigor
- Can't just eyeball results and say "this looks better"
- Need quantitative evidence to support claims
- Statistical validation is expected in research publications

### Reproducibility
- Other researchers can verify your findings
- Statistical tests provide objective criteria
- Transparent methodology builds credibility

### Thesis Quality
- Demonstrates understanding of research methodology
- Shows ability to apply statistical reasoning
- Meets academic standards for FYP/Honours projects

### Real-World Relevance
- Before deploying a robot, you need confidence it actually works
- Statistical validation reduces risk of deploying poorly-performing systems
- Industry expects evidence-based decision making

---

## 8. Quick Reference for Your Tests

### Running the Statistical Validation:

```bash
cd /home/aliriyazr1/NUbots_RL_FYP
source .venv/bin/activate

# Requires BOTH PPO and DDPG models
python -m pytest tests/integration/test_statistical_validation.py::TestAlgorithmComparison::test_ppo_vs_ddpg_statistical_comparison \
  --ppo-model=path/to/ppo.zip \
  --ddpg-model=path/to/ddpg.zip \
  -v -s
```

### Output Interpretation:

Look for:
1. **p-value < 0.05** → Significant difference exists
2. **Cohen's d ≥ 0.5** → Difference is meaningful in practice
3. **95% CI doesn't include 0** → Confirms significance

### For Your Thesis:

Report all of:
- Means and standard deviations for both algorithms
- t-statistic and degrees of freedom
- p-value (exact value, not just "< 0.05")
- Effect size (Cohen's d) with interpretation
- 95% confidence interval for mean difference

---

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- Henderson, P., et al. (2018). Deep reinforcement learning that matters. *AAAI*.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

---

**Summary:** These statistical tests give you the **evidence** to confidently state whether your algorithms differ in performance, and by how much. This is essential for academic research and real-world deployment!
