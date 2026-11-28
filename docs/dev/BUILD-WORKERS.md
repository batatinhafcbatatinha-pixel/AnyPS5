# Build workers

How many compile jobs a build runs at once, and what the build tools read to decide it.

| Read | From | Meaning |
| --- | --- | --- |
| `JOBS=N` | environment of `build-runtime.sh` | Exactly N jobs. |
| `PORT_JOBS` | environment, set by the launcher's setup for every build step | The default when `JOBS` is not set. |
| `PORT_WORKERS_AUTO=1` with a jobserver in `MAKEFLAGS` | environment, set by the launcher's worker monitor | The build is started without a jobs option: Ninja takes its jobs from the jobserver, and the monitor adds or removes jobs while the build runs, following the free RAM and the CPU. |

The monitor and the launcher's setting (Settings > Recompilation workers: automatic, recompiler default, or a fixed
number) live in port-installer (`tools/port-installer/setup/port_workers.py`, described in its
`docs/DEVELOPING.md`). Without the launcher nothing changes: the script uses `JOBS` or every CPU.

Change here when a build command is added: the only build the launcher starts here is `tools/ps5port/build-runtime.sh` (the relinker and the system libraries from source, where no release is used); it sets its jobs option in the three lines after `jobs=`. Converting a dump (`tools/ps5port/ps5port.py build`) is one run of the relinker, a single process with no jobs option, so workers do not change it. Every `cmake --build` there takes the option from `$parallel`;
a new build command must do the same, or the monitor cannot move its jobs.
