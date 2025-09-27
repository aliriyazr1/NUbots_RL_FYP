# Claude Code Instructions for Soccer RL FYP Project

## Project Context
**Student:** Ali Riyaz (C3412624)  
**Project:** Reinforcement Learning for Soccer-Playing Robots  
**Environment:** SoccerEnv (2D soccer simulation)  
**Algorithms:** PPO, DDPG  
**Current Status:** Movement smoothing implemented, reward function needs optimisation  

## Core Requirements

### 1. Code Modification Philosophy
- **NEVER create new files** - Always edit existing code in place unless it makes sense to create a new file
- **NEVER create new functions** unless specifically requested - Modify existing functions
- **ALWAYS explain the solution** before implementing - Include reasoning and sources
- **THINK HARD** before responding - Plan the solution thoroughly
- **FIX SPECIFIC PROBLEMS** - Don't rewrite entire systems

### 2. Academic Standards
- Use **Australian spelling** (behaviour, optimise, realise, colour, etc.)
- **Cite sources** for any algorithms, techniques, or approaches used
- **Explain research contribution** - How does this advance the field?
- **Document testing methodology** - Reference testing plan where relevant
- **Academic writing style** - Suitable for research report and presentation

### 3. Git/Repository Rules
- **NEVER** use git commit, git push, or create pull requests
- **NEVER** modify .git files or repository metadata
- Only edit source code files when requested

## Key Problems to Address

### 1. Reward Function Issues
**Current Problem:** Agent learns "technically correct but wrong" behaviours (reward: 33,402)
**Location:** `soccerenv.py` - `_calculate_reward()` method
**Solution Approach:**
- Multi-component reward design (possession + goal progress + opponent avoidance)
- Anti-exploitation measures (stationary penalties, boundary violations)
- Potential-based shaping for consistent learning signals
- Mathematical functions: sigmoid (smooth transitions), gaussian (optimal zones), exponential (critical behaviours)

### 2. Testing/Evaluation Framework
**Requirements from Testing Plan:**
- Unit testing for reward functions (edge cases)
- Performance benchmarking (RL vs rule-based)
- Statistical validation (convergence, stability)
- Physics validation (collision detection, ball dynamics)

## Specific Instructions for Common Tasks

### Reward Function Modification
```
TASK: Fix reward function to prevent exploitation

APPROACH:
1. Analyse current reward components in _calculate_reward()
2. Identify exploitation opportunities (stationary behaviour, boundary camping)
3. Design multi-objective reward with proper mathematical functions
4. Implement temporal penalties for prolonged bad behaviours
5. Add debug logging for reward component analysis


EXPLANATION REQUIRED:
- Why each reward component prevents specific exploitations
- Mathematical justification for function choices
- Expected impact on learning behaviour
- Walkthrough different possible scenarios and outcomes that the agent can take to account for every action possible to better design the function to avoid the situation where it technically maximises rewar but learns wrong behaviours that cannot solve the problem
```

### Testing Implementation
```
TASK: Add test cases for reward function

APPROACH:
1. Identify edge cases from testing plan
2. Create test scenarios within existing test framework
3. Validate reward bounds and scaling
4. Test anti-exploitation measures
5. Generate test reports for academic documentation

TEST SCENARIOS:
- Robot stationary with ball (should decrease reward over time)
- Robot approaching goal optimally (should increase exponentially)
- Robot avoiding opponent (should use gaussian optimal distance)
- Boundary/wall exploitation attempts (should be penalised)

ACADEMIC VALUE:
- Demonstrates systematic testing methodology
- Validates theoretical reward design
- Provides quantitative evidence for thesis
```

### Data Analysis/Plotting
```
TASK: Create reusable plotting function for training analysis

REQUIREMENTS:
- Work with both PPO and DDPG log formats
- Generate publication-quality figures
- Support evaluations.npz and TensorBoard data
- Include statistical analysis (mean, std, confidence intervals)
- Export data for academic writing

ACADEMIC FOCUS:
- Learning curve analysis for methodology validation
- Algorithm comparison for research contribution
- Performance metrics aligned with testing plan
- Statistical significance testing
```

## Response Format Requirements

### 1. Planning Phase
```
## Problem Analysis
[Identify specific issue and location in code]

## Research Context  
[Cite relevant academic sources and techniques]

## Solution Design
[Explain approach with mathematical/theoretical justification]

## Expected Outcome
[Predict how this fixes the problem and contributes to research]

## Implementation Plan
[Step-by-step modification approach]
```

### 2. Implementation Phase
```
## Code Modifications
[Show only the specific lines being changed, with clear before/after]

## Explanation
[Why these changes solve the problem]

## Testing Strategy
[How to validate the fix works]

## Academic Contribution
[How this advances the research/field]
```

## Academic Writing Support

### Research Contribution Language
- "This implementation extends the work of [Author] by..."
- "Drawing on the theoretical framework of [Theory], we implement..."
- "Following the methodology established by [Research], our approach..."
- "This addresses the limitation identified in [Previous Work]..."

### Technical Implementation Language
- "The reward function incorporates potential-based shaping (Ng et al., 1999) to..."
- "Following Schulman et al.'s PPO algorithm, we modify the policy update to..."
- "Using the evaluation framework proposed in [Testing Literature]..."

### Results Documentation Language
- "Statistical analysis reveals significant improvement (p < 0.05) in..."
- "Convergence analysis demonstrates stable learning after X episodes..."
- "Performance benchmarking indicates Y% improvement over baseline..."

## Quick Reference Commands

### For Code Analysis
```
"Analyse the reward function in _calculate_reward() method. Identify potential exploitation issues and explain mathematical approach to fix them. Cite relevant reward shaping literature."
```

### For Testing
```
"Add edge case tests for reward function based on testing plan requirements. Focus on anti-exploitation validation. Include statistical analysis for academic documentation."
```

### For Data Analysis
```
"Create reusable plotting function for training analysis. Must work with evaluations.npz format and generate publication-quality figures for research report."
```

### For Implementation
```
"Modify existing [specific function] to implement [specific solution]. Explain theoretical basis and expected research contribution. Include academic sources used."
```

## Success Criteria

### Code Quality
- ✅ Fixes specific problem without breaking existing functionality
- ✅ Includes proper documentation and comments
- ✅ Follows existing code style and patterns
- ✅ Adds appropriate error handling

### Academic Quality
- ✅ Cites relevant research sources
- ✅ Explains theoretical foundation
- ✅ Documents methodology clearly
- ✅ Provides quantitative validation

### Testing Quality
- ✅ Covers all edge cases identified in testing plan
- ✅ Includes statistical validation
- ✅ Generates data suitable for academic reporting
- ✅ Demonstrates systematic evaluation approach

Remember: Quality over quantity. Better to have one well-implemented, academically sound solution than multiple untested features.