# Gunfolds v2.0 Documentation Index

**Complete Guide to All Documentation Files**

---

## 📚 Quick Navigation

| Your Situation | Start Here | Time Required |
|----------------|-----------|---------------|
| **New to gunfolds** | [docs/QUICKSTART.md](docs/QUICKSTART.md) | 15 minutes |
| **Migrating from old code** | [MIGRATION.md](MIGRATION.md) | 30-60 minutes |
| **Need quick script lookup** | [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md) | 2 minutes |
| **Visual migration guide** | [MIGRATION_WORKFLOW.md](MIGRATION_WORKFLOW.md) | 10 minutes |
| **Overview of changes** | [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) | 15 minutes |
| **Tune hyperparameters** | [docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md) | 30 minutes |
| **Use cluster/SLURM** | [docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md) | 30 minutes |
| **Documentation hub** | [docs/README.md](docs/README.md) | 10 minutes |

---

## 📖 All Documentation Files

### Core Migration Documents

#### 1. [MIGRATION.md](MIGRATION.md)
**The complete migration guide**

**Length**: ~1,500 lines  
**Read time**: 30-60 minutes (or use as reference)

**What's inside**:
- ✅ Overview of refactoring (why, what changed, benefits)
- ✅ Quick reference tables (all scripts, old → new)
- ✅ Detailed category-by-category migration (8 categories)
- ✅ 10+ command-line examples (before/after comparisons)
- ✅ Breaking changes (imports, function signatures, naming)
- ✅ New directory structure
- ✅ 10 FAQ with detailed answers
- ✅ Migration checklist
- ✅ Timeline & deprecation schedule

**Best for**: Thorough understanding of migration

**Example use cases**:
- Understanding what changed and why
- Migrating multiple scripts
- Updating automation workflows
- Learning about new features

---

#### 2. [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md)
**Quick lookup table: A-Z script mapping**

**Length**: ~800 lines  
**Read time**: 2-5 minutes (lookup), 15 minutes (complete read)

**What's inside**:
- ✅ Complete A-Z alphabetical listing of ALL old scripts
- ✅ Direct mapping: old script → new script + exact command
- ✅ Category-based quick reference tables
- ✅ 5 detailed migration examples
- ✅ Parameter name changes
- ✅ Import path changes
- ✅ Scripts moved to legacy/
- ✅ Verification instructions

**Best for**: Quick lookups ("Where is my script?")

**Example use cases**:
- Finding new location of specific old script
- Quick command reference
- Updating single script usage

---

#### 3. [MIGRATION_WORKFLOW.md](MIGRATION_WORKFLOW.md)
**Visual step-by-step migration guide**

**Length**: ~500 lines  
**Read time**: 10-20 minutes

**What's inside**:
- ✅ Visual flowcharts for different user types
- ✅ Quick migration path (5-10 min)
- ✅ Thorough migration path (30-60 min)
- ✅ Complete migration path (2-4 hours)
- ✅ Cluster user specific path
- ✅ Decision trees
- ✅ Verification checklists
- ✅ Progress tracker
- ✅ Troubleshooting decision tree

**Best for**: Visual learners, step-by-step guidance

**Example use cases**:
- Choosing your migration approach
- Following a structured migration process
- Tracking migration progress

---

#### 4. [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
**Overview of the refactoring project**

**Length**: ~700 lines  
**Read time**: 15 minutes

**What's inside**:
- ✅ Documentation package overview
- ✅ Visual structure comparison (before/after)
- ✅ Consolidation statistics
- ✅ Migration paths by user type
- ✅ Key features of each document
- ✅ Example use cases
- ✅ Migration checklist
- ✅ Benefits summary
- ✅ Contact & support info

**Best for**: Understanding the big picture

**Example use cases**:
- Overview before diving into details
- Understanding scope of changes
- Presenting to team/stakeholders
- Deciding migration approach

---

### Getting Started Documents

#### 5. [docs/QUICKSTART.md](docs/QUICKSTART.md)
**Getting started with gunfolds (new or migrating users)**

**Length**: ~600 lines  
**Read time**: 15-30 minutes

**What's inside**:
- ✅ Installation instructions
- ✅ Basic concepts (causal discovery, key components)
- ✅ Common tasks (5 categories with examples)
- ✅ Script organization guide
- ✅ Complete first analysis walkthrough (5 steps)
- ✅ Command-line help usage
- ✅ Configuration files
- ✅ Batch processing
- ✅ Cluster computing intro
- ✅ Troubleshooting
- ✅ Quick command reference

**Best for**: First-time users, learning the basics

**Example use cases**:
- Installing gunfolds
- Running first analysis
- Understanding script organization
- Learning common workflows

---

#### 6. [docs/README.md](docs/README.md)
**Documentation hub with links to all guides**

**Length**: ~500 lines  
**Read time**: 10 minutes

**What's inside**:
- ✅ Documentation overview & quick links
- ✅ Guides for new users
- ✅ Migration quick reference
- ✅ New directory structure explained
- ✅ Common workflows (3 complete examples)
- ✅ Key improvements in v2.0
- ✅ Script categories explained (6 categories)
- ✅ Getting help resources
- ✅ FAQ
- ✅ Contact information

**Best for**: Starting point for all documentation

**Example use cases**:
- Finding the right document
- Understanding doc structure
- Quick overview of capabilities

---

### Advanced Topics Documents

#### 7. [docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md)
**Comprehensive hyperparameter tuning guide**

**Length**: ~800 lines  
**Read time**: 30-60 minutes

**What's inside**:
- ✅ Overview of hyperparameter tuning
- ✅ VAR model parameters (order, regularization, coefficients, thresholds)
- ✅ RASL solver parameters (depth, timeout, optimization method)
- ✅ Systematic tuning workflow (4 steps)
- ✅ Analysis tools and metrics
- ✅ Visualization (parameter effects, learning curves, heatmaps)
- ✅ Best practices (6 guidelines)
- ✅ 3 complete example tuning studies
- ✅ Advanced topics (multi-objective, Bayesian optimization, transfer learning)

**Best for**: Optimizing performance, systematic parameter search

**Example use cases**:
- Tuning VAR model for your data
- Optimizing RASL solver settings
- Running systematic grid search
- Understanding parameter effects

---

#### 8. [docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md)
**SLURM/cluster computing guide**

**Length**: ~700 lines  
**Read time**: 30-60 minutes

**What's inside**:
- ✅ Why use a cluster
- ✅ SLURM basics (concepts, access, setup)
- ✅ Submitting jobs (2 methods: built-in & manual)
- ✅ SLURM directives explained (complete table)
- ✅ Monitoring and managing jobs
- ✅ Collecting results
- ✅ Best practices (7 tips)
- ✅ 4 common use cases with complete examples
- ✅ Troubleshooting (6 problems + solutions)
- ✅ Advanced topics (dependencies, GPU, interactive sessions)

**Best for**: Running large-scale experiments on clusters

**Example use cases**:
- Submitting SLURM jobs
- Batch processing
- Large-scale benchmarks
- Hyperparameter grid search on cluster

---

### Technical Documents

#### 9. [code.plan.md](code.plan.md)
**Original refactoring plan (technical details)**

**Length**: ~280 lines  
**Read time**: 10-15 minutes

**What's inside**:
- ✅ Current issues identified (9 categories)
- ✅ Proposed new structure
- ✅ Implementation steps (3 phases)
- ✅ Expected benefits
- ✅ Risks & mitigation
- ✅ Timeline estimate
- ✅ To-do list

**Best for**: Understanding technical decisions, implementation details

**Example use cases**:
- Understanding design rationale
- Contributing to refactoring
- Implementing similar refactoring in other projects

---

## 🎯 Documentation by Use Case

### Use Case: "I'm new to gunfolds"

**Path**:
1. [docs/QUICKSTART.md](docs/QUICKSTART.md) - Learn basics (15 min)
2. [docs/README.md](docs/README.md) - Understand structure (10 min)
3. Try example workflows in QUICKSTART
4. Explore advanced guides as needed

---

### Use Case: "I used MVAR_fig4.py, where is it now?"

**Path**:
1. [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md) - Quick lookup (2 min)
   - Search for "MVAR_fig4"
   - Find: `benchmarks/benchmark_runner.py --method MVAR`
2. Done!

---

### Use Case: "I have many scripts to migrate"

**Path**:
1. [MIGRATION_WORKFLOW.md](MIGRATION_WORKFLOW.md) - Choose approach (5 min)
2. [MIGRATION.md](MIGRATION.md) - Detailed migration (30-60 min)
3. [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md) - Reference as needed
4. [docs/QUICKSTART.md](docs/QUICKSTART.md) - Learn new features (15 min)

---

### Use Case: "I need to optimize parameters"

**Path**:
1. [docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md) - Complete guide (30-60 min)
2. [docs/QUICKSTART.md](docs/QUICKSTART.md) - Task 5 example (5 min)
3. Run tuning workflow

---

### Use Case: "I need to run jobs on cluster"

**Path**:
1. [docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md) - Complete guide (30-60 min)
2. [MIGRATION_WORKFLOW.md](MIGRATION_WORKFLOW.md) - Cluster user path (15 min)
3. Test with small job
4. Scale up

---

### Use Case: "I need to understand what changed"

**Path**:
1. [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Overview (15 min)
2. [MIGRATION.md](MIGRATION.md) - Detailed changes (30 min)
3. [code.plan.md](code.plan.md) - Technical details (10 min)

---

### Use Case: "I have automation scripts to update"

**Path**:
1. [MIGRATION.md](MIGRATION.md) - Breaking changes section (10 min)
2. [MIGRATION_WORKFLOW.md](MIGRATION_WORKFLOW.md) - Complete migration path (follow steps)
3. [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md) - Reference for each script
4. Test thoroughly

---

## 📊 Documentation Statistics

| Document | Lines | Primary Audience | Estimated Read Time |
|----------|-------|------------------|---------------------|
| MIGRATION.md | ~1,500 | Existing users | 30-60 min (or reference) |
| OLD_TO_NEW_SCRIPT_REFERENCE.md | ~800 | All users | 2 min (lookup) |
| MIGRATION_WORKFLOW.md | ~500 | Existing users | 10-20 min |
| REFACTORING_SUMMARY.md | ~700 | All users | 15 min |
| docs/QUICKSTART.md | ~600 | All users | 15-30 min |
| docs/README.md | ~500 | All users | 10 min |
| docs/experiments/hyperparameter_tuning.md | ~800 | Advanced users | 30-60 min |
| docs/experiments/cluster_guide.md | ~700 | Cluster users | 30-60 min |
| code.plan.md | ~280 | Developers | 10-15 min |
| **Total** | **~6,380 lines** | | |

---

## 🗺️ Documentation Map

```
Gunfolds v2.0 Documentation
│
├── Entry Points
│   ├── DOCUMENTATION_INDEX.md (this file) ← START HERE
│   ├── docs/README.md ← Documentation hub
│   └── REFACTORING_SUMMARY.md ← Overview
│
├── Migration Guides
│   ├── MIGRATION.md ← Complete migration guide
│   ├── OLD_TO_NEW_SCRIPT_REFERENCE.md ← Quick lookup
│   └── MIGRATION_WORKFLOW.md ← Visual guide
│
├── Getting Started
│   ├── docs/QUICKSTART.md ← New user guide
│   └── docs/README.md ← Documentation hub
│
├── Advanced Topics
│   ├── docs/experiments/hyperparameter_tuning.md ← Tuning
│   └── docs/experiments/cluster_guide.md ← SLURM/HPC
│
└── Technical
    └── code.plan.md ← Implementation details
```

---

## 🚀 Recommended Reading Paths

### Path 1: New User (Never used gunfolds)

```
1. docs/README.md (10 min)
   ↓
2. docs/QUICKSTART.md (15-30 min)
   ↓
3. Try example workflow
   ↓
4. Explore advanced guides as needed
```

**Total time**: 30-45 minutes to get started

---

### Path 2: Casual Migrating User (Occasional script usage)

```
1. REFACTORING_SUMMARY.md (15 min)
   ↓
2. OLD_TO_NEW_SCRIPT_REFERENCE.md (2-5 min per script)
   ↓
3. Test new commands
```

**Total time**: 20-30 minutes

---

### Path 3: Regular Migrating User (Frequent usage)

```
1. REFACTORING_SUMMARY.md (15 min)
   ↓
2. MIGRATION.md - Overview & Quick Reference (15 min)
   ↓
3. MIGRATION.md - Detailed sections for your scripts (20-30 min)
   ↓
4. docs/QUICKSTART.md - New features (15 min)
   ↓
5. Update workflows & test
```

**Total time**: 1-2 hours

---

### Path 4: Power User (Custom workflows, automation)

```
1. REFACTORING_SUMMARY.md (15 min)
   ↓
2. MIGRATION.md - Complete read (30-60 min)
   ↓
3. MIGRATION_WORKFLOW.md - Complete path (15 min)
   ↓
4. code.plan.md (10 min)
   ↓
5. Update all workflows
   ↓
6. docs/experiments/hyperparameter_tuning.md (30 min)
   ↓
7. Optimize workflows
```

**Total time**: 2-4 hours

---

### Path 5: Cluster User

```
1. REFACTORING_SUMMARY.md (15 min)
   ↓
2. MIGRATION.md - SLURM sections (15 min)
   ↓
3. docs/experiments/cluster_guide.md (30-60 min)
   ↓
4. MIGRATION_WORKFLOW.md - Cluster path (15 min)
   ↓
5. Update SLURM scripts
   ↓
6. Test & scale up
```

**Total time**: 1-2 hours

---

## 📝 Document Features Summary

### MIGRATION.md
- ✅ Most comprehensive
- ✅ 8 categories detailed
- ✅ 10+ examples
- ✅ Breaking changes
- ✅ FAQ

### OLD_TO_NEW_SCRIPT_REFERENCE.md
- ✅ A-Z complete listing
- ✅ Fastest lookup
- ✅ Direct commands
- ✅ Category tables
- ✅ Examples

### MIGRATION_WORKFLOW.md
- ✅ Visual flowcharts
- ✅ Step-by-step paths
- ✅ Progress tracker
- ✅ Decision trees
- ✅ Checklists

### REFACTORING_SUMMARY.md
- ✅ Big picture overview
- ✅ Statistics
- ✅ Visual comparison
- ✅ Benefits explained
- ✅ User paths

### docs/QUICKSTART.md
- ✅ Installation
- ✅ Basic concepts
- ✅ Common tasks
- ✅ First workflow
- ✅ Troubleshooting

### docs/experiments/hyperparameter_tuning.md
- ✅ All parameters
- ✅ Tuning workflow
- ✅ 3 examples
- ✅ Best practices
- ✅ Advanced topics

### docs/experiments/cluster_guide.md
- ✅ SLURM basics
- ✅ 2 job methods
- ✅ 4 use cases
- ✅ Troubleshooting
- ✅ Advanced topics

---

## 🎓 Learning Objectives by Document

| If you want to... | Read this document |
|-------------------|-------------------|
| Learn gunfolds from scratch | docs/QUICKSTART.md |
| Find where old script went | OLD_TO_NEW_SCRIPT_REFERENCE.md |
| Understand migration completely | MIGRATION.md |
| Follow visual migration steps | MIGRATION_WORKFLOW.md |
| Get overview of changes | REFACTORING_SUMMARY.md |
| Optimize parameters | docs/experiments/hyperparameter_tuning.md |
| Use cluster effectively | docs/experiments/cluster_guide.md |
| Understand technical decisions | code.plan.md |
| Navigate all docs | docs/README.md |

---

## ✅ Documentation Quality Checklist

All documents include:
- ✅ Clear table of contents
- ✅ Time estimates
- ✅ Target audience
- ✅ Practical examples
- ✅ Code blocks with syntax highlighting
- ✅ Cross-references to other docs
- ✅ Quick reference sections
- ✅ Troubleshooting (where applicable)
- ✅ Contact/support info
- ✅ Version and date

---

## 🔗 External Resources

- **GitHub Repository**: https://github.com/your-org/gunfolds
- **Issues**: https://github.com/your-org/gunfolds/issues
- **Discussions**: https://github.com/your-org/gunfolds/discussions
- **API Documentation**: (Sphinx docs if available)

---

## 🆘 Still Can't Find What You Need?

1. **Use Ctrl+F / Cmd+F** in this index to search for keywords
2. **Check [docs/README.md](docs/README.md)** for documentation hub
3. **Search [MIGRATION.md](MIGRATION.md)** FAQ section
4. **Open GitHub Issue** with tag `documentation`
5. **Ask in GitHub Discussions**

---

## 📞 Support & Feedback

### Documentation Feedback

Found an issue with documentation?
- Typos, errors: Open GitHub issue with tag `documentation`
- Missing information: Open GitHub issue with tag `documentation-enhancement`
- Unclear sections: Comment in GitHub Discussions

### General Support

- **Questions**: GitHub Discussions
- **Bugs**: GitHub Issues
- **Feature requests**: GitHub Issues with tag `enhancement`

---

**This index was last updated**: December 2, 2025  
**Gunfolds version**: 2.0.0  
**Total documentation size**: ~6,380 lines across 9 files  

**Happy analyzing!** 🚀

---

## Quick Start Reminder

```
New user?          → docs/QUICKSTART.md
Need quick lookup? → OLD_TO_NEW_SCRIPT_REFERENCE.md
Migrating?         → MIGRATION.md
Visual guide?      → MIGRATION_WORKFLOW.md
Overview?          → REFACTORING_SUMMARY.md
```

