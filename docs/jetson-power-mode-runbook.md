# Jetson power-mode validation

Rakshak Lens performance tests are invalid until the Jetson has a compatible,
explicit NVIDIA power profile. A mismatched profile can fail during boot after
partially changing clocks or CPU availability, leaving the device slower than
its hardware specification without making the application unhealthy.

## Read-only gate

Run this before every Jetson benchmark and after every JetPack or module change:

```bash
python3 scripts/jetson_power_mode_doctor.py --json
```

For a MAXN capacity test, require both the mode and all physical CPUs:

```bash
python3 scripts/jetson_power_mode_doctor.py \
  --require-mode MAXN \
  --require-all-cpus
```

The doctor fails when the active profile references nonexistent CPUs, nvpmodel
has no active mode, the query fails, or an explicitly required capacity gate is
not met. It is intentionally read-only.

## Selecting a profile

Do not select a profile from CPU count alone. Confirm the physical module SKU,
RAM size, carrier-board compatibility, cooling, and available power supply.
The doctor lists installed profiles whose CPU references match the running
kernel, but an operator must still confirm the exact SKU.

For the observed 6-core / 8 GB Orin NX staging unit, the installed 6-core
profile is:

```text
/etc/nvpmodel/nvpmodel_p3767_0001.conf
```

The prior `p3767_0000` link referenced CPU 6 and CPU 7, which do not exist on
that unit. The boot service consequently exited with status 255 and left power
mode unset.

## Apply on the observed staging unit

This operation requires root and reboots the Jetson. Keep an SSH rollback path
and do it only during a maintenance window.

```bash
sudo ln -s "$(readlink -f /etc/nvpmodel.conf)" \
  /etc/nvpmodel.conf.pre-rakshak
sudo ln -sfn /etc/nvpmodel/nvpmodel_p3767_0001.conf /etc/nvpmodel.conf
sudo nvpmodel --force -m 0
```

After reboot, verify the doctor, Docker health, both camera streams, temperature,
and input power under a representative inference soak. MAXN should not be used
without adequate cooling and power headroom.

## Roll back to the correct lower-power profile

Mode 2 is the 15 W mode in this unit's `p3767_0001` profile. Switching back may
also reboot because GPU power-gating changes.

```bash
sudo nvpmodel --force -m 2
python3 scripts/jetson_power_mode_doctor.py --require-mode 15W
```

Do not restore the incompatible `p3767_0000` symlink as a performance rollback;
that restores the boot failure rather than a valid lower-power mode.
