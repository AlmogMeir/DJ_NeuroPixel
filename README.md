# DJ_NeuroPixel

DataJoint pipeline for behavioral (BPOD) and electrophysiology (Neuropixel) data processing.

## Prerequisites

- Python 3.8+
- DataJoint: `pip install datajoint pandas numpy scipy matplotlib`
- LAB & EXP schemas created in database (via MATLAB)
- Session records must exist before populating data

## Database Configuration

**Each user must configure their own credentials.**

Create `dj_local_conf.json` in the project root:

```json
{
  "database.host": "your-database-host",
  "database.user": "your-username",
  "database.password": "your-password"
}
```

**Note**: This file is git-ignored. Never commit credentials to the repository.

---

## BPOD Pipeline

Process behavioral data from BPOD maze experiments.

### Files
- `mazeBPOD.py` - Schema and table definitions
- `populate.py` - Data extraction and population
- `drop_tables.py` - Remove tables from database

### Tables
- **SessionTrial**: Trial numbers and start times
- **Port**: Port configurations (port number, reward size)
- **Block**: Block-port associations
- **Reward**: Reward events (size, timing, depletion)
- **LickEvent**: Lick events (timing, duration, early detection)

### Usage

**1. Configure in `populate.py`:**
```python
subject_id = 105101
session = 2
matlab_file = "your_data_file.mat"
```

**2. Run:**
```bash
cd BPOD
python populate.py
```

**Output:**
- CSV files in `output/YYYYMMDD/` (date from filename)
- Data inserted to DataJoint tables

**3. Drop tables (optional):**
```bash
python drop_tables.py
```

### Features
- Automatic SessionTrial creation from trial count
- IRI-based early lick detection
- Reward depletion tracking
- Multi-port trial support
- Sensor noise filtering

---

## NeuropixelEphys Pipeline

Process Neuropixel electrophysiology data.

### Files
- `ArsenyEPHYS.py` - Schema and table definitions
- `populate.py` - Data extraction and population
- `drop_tables.py` - Remove tables from database
- `dj_connect.py` - Database connection utilities
- `getSchema.py` - Schema retrieval utilities

### Tables
- **Probe**: Probe information and geometry
- **ElectrodeGroup**: Electrode groupings
- **CellType**: Cell type classifications
- **Unit**: Sorted units with quality metrics
- **UnitQualityType**: Quality metric definitions
- **TrialSpikes**: Spike times per trial per unit

### Usage

**1. Update paths in `populate.py`:**
```python
# Set paths to Kilosort output and probe geometry
```

**2. Run:**
```bash
cd NeuropixelEphys
python populate.py
```

**3. Drop tables (optional):**
```bash
python drop_tables.py
```

### Data Sources
- Kilosort spike sorting output
- Probe geometry files
- Trial alignment data

---

## Troubleshooting

**Connection Error:**
```
Error: Access denied for user
```
→ Check `dj_local_conf.json` credentials

**Foreign Key Error:**
```
Error: Cannot add or update a child row
```
→ Ensure parent session exists in database

**Import Error:**
```
ImportError: No module named 'datajoint'
```
→ Run: `pip install datajoint pandas numpy scipy matplotlib`

---

## Data Flow

### BPOD
```
MATLAB .mat file
  → SessionTrial (trial count)
  → Port (AllRewardSizes)
  → Block (Blocks array)
  → Reward (States.RewardX)
  → LickEvent (Events.PortXIn/Out)
```

**Insertion order**: SessionTrial → Port → Block → Reward → LickEvent

### Ephys
```
Kilosort output
  → Unit (cluster info)
  → TrialSpikes (spike times)
  → Probe geometry
```
