---
name: thermal_management
description: Monitor and manage M4 MacBook Pro thermal state during training. Use before and during any model training, batch size discovery, or compute-intensive operation. Enforces temperature thresholds and pause protocols to prevent thermal damage.
---

# Thermal Management Skill

Monitor and manage hardware thermals during compute-intensive work.

## When to Use

- Before starting any training run
- During training (periodic checks)
- Before batch size discovery
- Before any HPO sweep
- When system feels hot or fans are loud
- User says "check thermals", "temperature check"

## Hardware Context

**Machine:** M4 MacBook Pro, 128GB Unified Memory
**Cooling:** Basement environment (50-60°F ambient)
**Advantage:** Cold ambient allows sustained compute

## Temperature Thresholds

| Range | Status | Action |
|-------|--------|--------|
| <70°C | 🟢 Normal | Full operation permitted |
| 70-85°C | 🟡 Acceptable | Continue with monitoring every 10 min |
| 85-95°C | 🟠 Warning | Consider pause, reduce batch size |
| >95°C | 🔴 Critical | **IMMEDIATE STOP** |

## Monitoring Commands

### Check Temperature

```bash
# Primary method - requires sudo
sudo powermetrics --samplers smc -i 1000 -n 1 | grep -i "temp"

# Alternative - if powermetrics unavailable
# Check Activity Monitor > CPU > Temperature
```

### Continuous Monitoring

```bash
# Log temperature every 30 seconds during training
while true; do
  echo "$(date): $(sudo powermetrics --samplers smc -i 1000 -n 1 | grep 'CPU die temperature')"
  sleep 30
done
```

## Pre-Training Checklist

Before starting any training run:

1. **Check current temperature**
   ```bash
   sudo powermetrics --samplers smc -i 1000 -n 1 | grep -i temp
   ```

2. **Verify cooling setup**
   - Laptop on hard, elevated surface
   - Adequate airflow around device
   - Basement environment confirmed

3. **Note starting conditions**
   - Current temp: ___°C
   - Ambient temp: ~___°F
   - Expected duration: ___ hours

4. **Set monitoring interval**
   - Short runs (<1 hr): Check at end
   - Medium runs (1-4 hr): Check every 30 min
   - Long runs (>4 hr): Check every 15 min

## During Training Protocol

### Routine Check (Every N Minutes)

```markdown
## Thermal Check - [TIME]

- CPU temp: [X]°C
- Status: [🟢/🟡/🟠/🔴]
- Action: [Continue/Monitor/Reduce/STOP]
- Training progress: [epoch/step]
```

### If Temperature Rising

1. **70-85°C**: Note it, continue monitoring more frequently
2. **85-95°C**: 
   - Pause training if safe checkpoint available
   - Reduce batch size by 50%
   - Allow 5-10 min cooldown
   - Resume with smaller batch
3. **>95°C**:
   - **STOP IMMEDIATELY**
   - Kill training process
   - Allow full cooldown (15-20 min)
   - Report to user before resuming

## Batch Size and Thermals

Larger batch sizes = higher thermal load

| Param Budget | Starting Batch | Thermal Watch |
|--------------|----------------|---------------|
| 2M | 64-128 | Low risk |
| 20M | 32-64 | Medium risk |
| 200M | 8-16 | High risk - monitor closely |

If running hot:
- Reduce batch size by 50%
- Gradient accumulation can compensate for effective batch size

## Emergency Stop Procedure

If temperature exceeds 95°C:

1. **Kill the process**
   ```bash
   # Find and kill Python training process
   pkill -f "python.*train"
   ```

2. **Verify stopped**
   ```bash
   ps aux | grep python
   ```

3. **Wait for cooldown**
   - Target: <70°C before resuming
   - Typically 15-20 minutes

4. **Report to user**
   ```
   🔴 THERMAL EMERGENCY STOP
   
   Temperature exceeded 95°C at [TIME]
   Training stopped at [epoch/step]
   Checkpoint: [path if saved]
   
   Waiting for cooldown before resume.
   Current temp: [X]°C
   ```

5. **Resume with adjustments**
   - Reduce batch size
   - Consider shorter training windows
   - Increase monitoring frequency

## Output Formats

### Pre-Training Report

```
🌡️ THERMAL CHECK - Pre-Training

Hardware: M4 MacBook Pro 128GB
Environment: Basement (~55°F ambient)
Current temp: [X]°C - 🟢 Normal

Training config:
- Param budget: [2M/20M/200M]
- Batch size: [N]
- Expected duration: [X] hours

Thermal risk: [Low/Medium/High]
Monitoring interval: Every [N] minutes

✅ Clear to start training
```

### Routine Check

```
🌡️ THERMAL CHECK - [TIME]

Temp: [X]°C - [🟢/🟡/🟠/🔴] [Status]
Training: Epoch [N], Step [M]
Action: [Continue/Adjust/Pause/STOP]
```

### Warning Event

```
⚠️ THERMAL WARNING

Temperature: [X]°C (threshold: 85°C)
Time: [TIME]
Training: Epoch [N], Step [M]

Action taken:
- [Reduced batch size / Paused / etc.]

Monitoring increased to every [N] minutes.
```

## Integration with Training Scripts

Training code should include thermal hooks:

```python
# Placeholder - actual implementation TBD
def check_thermal():
    """Check temperature and return status."""
    # Returns: 'normal', 'warning', 'critical'
    pass

def training_loop():
    for epoch in range(epochs):
        for batch in dataloader:
            # ... training step ...
            
            if step % thermal_check_interval == 0:
                status = check_thermal()
                if status == 'critical':
                    save_checkpoint()
                    raise ThermalEmergencyStop()
                elif status == 'warning':
                    log_thermal_warning()
```
