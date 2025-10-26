# 3-Way Policy Comparison

**Evaluation Date:** 2025-10-23 10:45:06
**Episodes:** 100
**Difficulty:** easy

---

## Individual Policy Performance

### DDPG Detailed Results

**Primary Metrics:**
- Goals Scored: 21/100
- Success Rate: 21.0%
- Average Reward: 12648.72 ± 3464.75
- Average Episode Length: 260.6 steps

**Detailed Performance Metrics:**
- Ball Possession Rate: 52.6%
- Robot Collision Rate: 19.0%
- Ball Out-of-Bounds Rate: 48.0%
- Mean Final Ball Distance: 1.62 meters

**Performance Assessment:**
- Status: DECENT - Success rate 21.0% shows learning
- Reward: EXCELLENT - Average reward 12648.7 is very good

---

### PPO Detailed Results

**Primary Metrics:**
- Goals Scored: 0/100
- Success Rate: 0.0%
- Average Reward: -563.31 ± 543.89
- Average Episode Length: 678.8 steps

**Detailed Performance Metrics:**
- Ball Possession Rate: 1.9%
- Robot Collision Rate: 25.0%
- Ball Out-of-Bounds Rate: 1.0%
- Mean Final Ball Distance: 3.01 meters

**Performance Assessment:**
- Status: NEEDS IMPROVEMENT - Success rate 0.0% is low
- Reward: NEGATIVE - Average reward -563.3, needs work

---

### Hand-Coded Detailed Results

**Primary Metrics:**
- Goals Scored: 0/100
- Success Rate: 0.0%
- Average Reward: -817.40 ± 772.75
- Average Episode Length: 747.3 steps

**Detailed Performance Metrics:**
- Ball Possession Rate: 2.2%
- Robot Collision Rate: 3.0%
- Ball Out-of-Bounds Rate: 1.0%
- Mean Final Ball Distance: 3.35 meters

**Performance Assessment:**
- Status: NEEDS IMPROVEMENT - Success rate 0.0% is low
- Reward: NEGATIVE - Average reward -817.4, needs work

---

## Comparison Table

| Metric | DDPG | PPO | Hand-Coded |
|--------|------|-----|------------|
| Mean Reward | 12648.72 ± 3464.75 | -563.31 ± 543.89 | -817.40 ± 772.75 |
| Median Reward | 12900.50 | -670.05 | -906.71 |
| Total Goals | 21 | 0 | 0 |
| Goals/Episode | 0.210 | 0.000 | 0.000 |
| Possession (%) | 52.6% | 1.9% | 2.2% |
| Collision Rate (%) | 19.0% | 25.0% | 3.0% |
| Out of Bounds (%) | 48.0% | 1.0% | 1.0% |
| Episode Length | 260.6 ± 229.6 | 678.8 ± 188.0 | 747.3 ± 104.8 |

## Winner Summary

- **Best Reward:** DDPG
- **Most Goals:** DDPG
- **Best Possession:** DDPG
- **Fewest Collisions:** Hand-Coded
