gunfolds
========

Tools to explore dynamic causal graphs in the case of  undersampled data helping to unfold the apparent structure into the underlying truth.

## 📢 Version 2.0 Released - Major Refactoring

**Gunfolds v2.0** features a complete reorganization for better maintainability and usability!

### 🆕 New Users
- **[Get Started →](docs/QUICKSTART.md)** - Quick start guide
- **[Documentation Hub →](docs/README.md)** - All guides and tutorials

### 🔄 Existing Users (Migrating from v1.x)
- **[Quick Script Lookup →](OLD_TO_NEW_SCRIPT_REFERENCE.md)** - Find where your script moved (2 min)
- **[Migration Guide →](MIGRATION.md)** - Complete migration instructions (30-60 min)
- **[Visual Workflow →](MIGRATION_WORKFLOW.md)** - Step-by-step visual guide

### 📚 All Documentation
- **[Documentation Index →](DOCUMENTATION_INDEX.md)** - Complete guide to all docs

**Key improvements in v2.0:**
- ✅ 118+ scripts consolidated to ~33 organized modules
- ✅ Eliminated 85+ duplicate files
- ✅ Clear folder structure (analysis/benchmarks/experiments/visualization)
- ✅ Unified interfaces with command-line parameters
- ✅ Comprehensive documentation
- ✅ Full backward compatibility (old scripts preserved in `scripts/legacy/`)

Documentation
===================
Please refer to the [Online Documentation](https://neuroneural.github.io/gunfolds/) for API reference and the documentation links above for v2.0 guides.

Installation
============

Install the gunfolds package

```bash
   pip install gunfolds
```

Additionally, install these packages to use gunfolds
   
graph-tool installation
-------------------------  
**1. Install** ``graph-tool``

To install ``graph-tool`` package with **conda install** run the following command

```bash
   conda install -c conda-forge graph-tool
```
   
To install ``graph-tool`` package with **brew install** run the following command

```bash
   brew install graph-tool
```

PyGObject installation
-------------------------
**2. Install** ``PyGObject``

**This is only required if you need to use** ``gtool`` **module of the** ``gunfolds`` **package**

To install ``PyGObject`` package with **brew install** run the following command

```bash
   brew install pygobject3 gtk+3
```

To install ``PyGObject`` package in Windows, Linux and any other platforms please refer to the link

   https://pygobject.readthedocs.io/en/latest/getting_started.html

Acknowledgment
========
This work was initially supported by  NSF IIS-1318759 grant and is currently supported by NIH 1R01MH129047.
