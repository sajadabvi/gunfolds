# GSU URSA Cluster Adaptation Summary

## Overview

Adapted GCM parallel processing scripts to match the GSU URSA cluster configuration, following the format of `slurm_hyperparameter_tuning.sh`.

---

## ✅ Changes Made

### 1. **`slurm_gcm_parallel.sh` - Complete Rewrite**

#### SLURM Directives (GSU URSA Format)
```bash
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 task  
#SBATCH -c 1                    # 1 CPU per task
#SBATCH --mem=4g                # 4GB memory
#SBATCH -p qTRDGPU              # qTRDGPU partition
#SBATCH -t 7200                 # 2 hours (in seconds)
#SBATCH -J gcm_parallel         # Job name
#SBATCH -e ./err/gcm_error%A-%a.err
#SBATCH -o ./out/gcm_out%A-%a.out
#SBATCH -A psy53c17             # Account
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe
```

**Key Changes:**
- ✅ Added account (`-A psy53c17`)
- ✅ Added partition (`-p qTRDGPU`)
- ✅ Added email notifications
- ✅ Changed output paths to `./err/` and `./out/`
- ✅ Added `--oversubscribe` flag
- ✅ Changed time format to seconds

#### Environment Setup
```bash
# Module path (GSU URSA specific)
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

# Conda activation (your account)
source /home/users/mabavisani/anaconda3/bin/activate
conda activate multi_v3

# OpenMP threads
export OMP_NUM_THREADS=1
```

#### Enhanced Logging
```bash
# Master log (created once)
logs/gcm_master_job_<JOB_ID>.log

# Task-specific logs (one per subject)
logs/gcm_task_<TASK_ID>.log
```

**Logging includes:**
- Start/end times
- Hostname
- Exit codes
- Success/failure status

#### Job Name Updates
```bash
# Dynamic job names
scontrol update jobid=$SLURM_JOB_ID name=gcm_${SUBJECT_IDX}
```

---

### 2. **`submit_gcm_parallel.sh` - Major Enhancement**

#### Better Directory Management
```bash
mkdir -p logs
mkdir -p err
mkdir -p out
```

#### Enhanced Metadata
```bash
# Comprehensive run metadata saved to:
gcm_roebroeck/<TIMESTAMP>/run_metadata.txt

# Includes:
- Timestamp
- Parameters (alpha, subjects, concurrent)
- Cluster configuration
- Output locations
- Job ID and array spec
```

#### Auto-Generated Progress Script
```bash
# Creates: logs/check_gcm_progress_<TIMESTAMP>.sh
# Quick status check for the specific run
```

#### Improved Output
```bash
# Cleaner, more informative output
- ✓ Success indicators
- ✗ Failure indicators
- Monitoring command examples
- Progress tracking commands
```

---

## 🔑 Key Differences from Generic Version

| Feature | Generic | GSU URSA |
|---------|---------|----------|
| **Partition** | Not specified | `qTRDGPU` |
| **Account** | Not specified | `psy53c17` |
| **Email** | Optional | `mabavisani@gsu.edu` |
| **Conda** | Generic path | `/home/users/mabavisani/anaconda3` |
| **Environment** | Generic | `multi_v3` |
| **Module Path** | Not set | GSU URSA specific path |
| **Output Dir** | `logs/` | `err/` and `out/` |
| **Logging** | Basic | Enhanced (master + task logs) |
| **Job Names** | Static | Dynamic per subject |

---

## 📋 File Comparison

### Before (Generic Cluster)

**`slurm_gcm_parallel.sh`:**
- Generic SLURM directives
- No account/partition
- No email notifications
- Simple logging to `logs/`
- No module loading
- Generic conda activation

**`submit_gcm_parallel.sh`:**
- Basic directory creation
- Simple metadata
- Basic monitoring commands

### After (GSU URSA Specific)

**`slurm_gcm_parallel.sh`:**
- GSU URSA SLURM directives
- Account: `psy53c17`
- Partition: `qTRDGPU`
- Email: `mabavisani@gsu.edu`
- Enhanced logging (master + task logs)
- GSU URSA module path
- Specific conda environment (`multi_v3`)
- Dynamic job naming

**`submit_gcm_parallel.sh`:**
- Creates `err/` and `out/` directories
- Comprehensive metadata file
- Auto-generated progress check script
- Enhanced monitoring commands
- Better error reporting

---

## 🎯 Matching `slurm_hyperparameter_tuning.sh`

The scripts now follow the same structure as your hyperparameter tuning script:

### Structure Match

| Component | Hyperparameter Tuning | GCM Parallel |
|-----------|----------------------|--------------|
| **SLURM Header** | ✓ Same format | ✓ Same format |
| **Account** | `psy53c17` | `psy53c17` |
| **Partition** | `qTRDGPU` | `qTRDGPU` |
| **Email** | `mabavisani@gsu.edu` | `mabavisani@gsu.edu` |
| **Output Dirs** | `./err/` and `./out/` | `./err/` and `./out/` |
| **Module Path** | GSU specific | GSU specific |
| **Conda** | `multi_v3` | `multi_v3` |
| **Logging** | Master + task logs | Master + task logs |
| **Job Naming** | Dynamic | Dynamic |

---

## 🚀 Usage on GSU URSA

### Quick Start

```bash
# 1. Navigate to scripts directory
cd /home/users/mabavisani/path/to/gunfolds/gunfolds/scripts

# 2. Make scripts executable (if needed)
chmod +x submit_gcm_parallel.sh slurm_gcm_parallel.sh

# 3. Test with 5 subjects
./submit_gcm_parallel.sh -n 5 -A 50 -c 2

# 4. Monitor
./logs/check_gcm_progress_<TIMESTAMP>.sh

# 5. Full run (after testing)
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
```

### Monitoring

```bash
# Check queue
squeue -u mabavisani

# Count progress
ls gcm_roebroeck/<TIMESTAMP>/csv/*Adj_labeled.csv | wc -l

# Watch mode
python analyze_gcm_results.py -t <TIMESTAMP> --watch --expected-subjects 50

# Check logs
tail -f out/gcm_out*.out
tail -f logs/gcm_task_0.log
```

---

## 📝 Configuration You May Need to Adjust

### In `slurm_gcm_parallel.sh`:

1. **Time Limit** (if subjects take longer)
   ```bash
   #SBATCH -t 10800  # 3 hours instead of 2
   ```

2. **Memory** (if OOM errors)
   ```bash
   #SBATCH --mem=8g  # 8GB instead of 4GB
   ```

3. **CPUs** (if using parallel bootstrap)
   ```bash
   #SBATCH -c 2      # 2 CPUs instead of 1
   export OMP_NUM_THREADS=2
   ```

4. **Email** (if you want different address)
   ```bash
   #SBATCH --mail-user=your.email@gsu.edu
   ```

### In `submit_gcm_parallel.sh`:

1. **Default Subjects** (line 8)
   ```bash
   MAX_SUBJECTS=50  # Change from 100 to 50
   ```

2. **Default Concurrent** (line 9)
   ```bash
   CONCURRENT=10    # Change from 20 to 10
   ```

---

## 🔍 Verification Steps

### Before First Run:

1. **Check Account Access:**
   ```bash
   sacctmgr show assoc user=mabavisani
   ```

2. **Verify Conda Environment:**
   ```bash
   source /home/users/mabavisani/anaconda3/bin/activate
   conda env list | grep multi_v3
   conda activate multi_v3
   python -c "import pandas; import numpy; print('OK')"
   ```

3. **Check Data File:**
   ```bash
   ls -lh ./fbirn/fbirn_sz_data.npz
   ```

4. **Check Partition:**
   ```bash
   sinfo -p qTRDGPU
   ```

5. **Test Single Subject (not array):**
   ```bash
   # Create test timestamp
   TIMESTAMP=$(date '+%m%d%Y%H%M%S')
   mkdir -p gcm_roebroeck/$TIMESTAMP/csv
   mkdir -p gcm_roebroeck/$TIMESTAMP/figures
   
   # Run single subject directly
   python gcm_on_ICA.py -S 0 -t $TIMESTAMP -A 50
   
   # Check output
   ls gcm_roebroeck/$TIMESTAMP/csv/
   ```

---

## 🆚 Side-by-Side Comparison

### SLURM Directives

**Generic:**
```bash
#SBATCH --job-name=gcm_parallel
#SBATCH --output=logs/gcm_%A_%a.out
#SBATCH --error=logs/gcm_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
```

**GSU URSA:**
```bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH -p qTRDGPU
#SBATCH -t 7200
#SBATCH -J gcm_parallel
#SBATCH -e ./err/gcm_error%A-%a.err
#SBATCH -o ./out/gcm_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe
```

---

## 📚 New Documentation

Created **`GCM_PARALLEL_GSU_URSA.md`**:
- GSU URSA specific instructions
- Cluster configuration details
- Troubleshooting for URSA
- Resource usage recommendations
- Quick reference commands

---

## ✨ Summary

### What Changed:
1. ✅ SLURM directives match GSU URSA cluster
2. ✅ Account and partition configured
3. ✅ Email notifications enabled
4. ✅ Environment setup matches hyperparameter script
5. ✅ Enhanced logging (master + task logs)
6. ✅ Better directory organization (`err/`, `out/`, `logs/`)
7. ✅ Auto-generated progress check scripts
8. ✅ Comprehensive metadata tracking

### What Stayed the Same:
1. ✅ Shared timestamp approach
2. ✅ Thread-safe directory creation
3. ✅ Analysis tool integration
4. ✅ Watch mode support
5. ✅ Flexible parameter configuration

---

## 🎉 Ready for GSU URSA!

The scripts are now configured to run on the GSU URSA cluster with the same format and conventions as your existing hyperparameter tuning jobs.

**Test first with 5 subjects, then scale up!**

---

## 📖 Documentation Files

1. **`GCM_PARALLEL_GSU_URSA.md`** - 📚 Complete guide for URSA (NEW)
2. **`GSU_URSA_ADAPTATION_SUMMARY.md`** - 📋 This file
3. **`GCM_PARALLEL_GUIDE.md`** - General parallel guide
4. **`PARALLEL_PROCESSING_SUMMARY.md`** - Overall summary

Use `GCM_PARALLEL_GSU_URSA.md` as your primary reference for running on URSA.

