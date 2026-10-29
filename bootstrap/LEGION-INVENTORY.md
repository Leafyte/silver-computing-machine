# Legion Hardware and Software Inventory

**Checked:** 2026-10-28 (Asia/Kolkata)  
**Host / Device Label:** `Klein` (Lenovo Legion Pro 5 16AFR10)  
**Checkout Path:** `C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main`  
**Execution Agent:** Antigravity  
**Authority:** Authorized bounded environment bootstrap under `AGENTS.md` and `prompts/ANTIGRAVITY-LEGION.md`

---

## 1. Operating System and Hardware

| Attribute | Specification | Evidence / Command Source |
|---|---|---|
| **OS / Edition** | Microsoft Windows 11 Home Single Language | `Get-CimInstance Win32_OperatingSystem` |
| **OS Version / Build** | 10.0.26200 Build 26200 (64-bit) | `[System.Environment]::OSVersion` |
| **Machine Architecture** | 64-bit AMD64 (x86_64) | `Win32_OperatingSystem.OSArchitecture` |
| **Device Model** | Lenovo Legion Pro 5 16AFR10 (System Model `83F2`) | `Get-CimInstance Win32_ComputerSystemProduct` |
| **CPU Model** | AMD Ryzen 9 9955HX | `Get-CimInstance Win32_Processor` |
| **CPU Cores / Threads** | 16 Physical Cores, 32 Logical Processors | `Win32_Processor.NumberOfCores` / `NumberOfLogicalProcessors` |
| **Physical Memory (RAM)** | 32.0 GB Total Physical RAM | `Win32_PhysicalMemory` |
| **Storage (System / Data)** | Drive `C:` (Windows-SSD), NTFS, 951.65 GB Total, **511.63 GB Free** | `Get-Volume` |
| **Storage (Recovery)** | Drive `WINRE_DRV`, NTFS, 1.95 GB Total, 1.08 GB Free | `Get-Volume` |

---

## 2. GPU and Accelerators

| Component | Value | Notes |
|---|---|---|
| **Discrete GPU** | NVIDIA GeForce RTX 5060 Laptop GPU | Active display controller |
| **GPU VRAM** | 8151 MiB (~8 GB GDDR6) | Reported by `nvidia-smi` |
| **Driver Version** | 610.88 (WDDM 32.0.16.1088) | Read-only observation via `nvidia-smi` and WMI |
| **CUDA Runtime / Driver** | CUDA UMD Version 13.3 | Visible via `nvidia-smi` |
| **NVCC Compiler** | Not on system PATH / Not installed | `nvcc` command not found; no drivers or toolkits altered |
| **Integrated GPU** | AMD Radeon(TM) 610M (512 MB AdapterRAM) | Secondary display controller |

> [!NOTE]
> In accordance with safety limits, no GPU drivers, CUDA toolkits, or global acceleration packages were installed or modified.

---

## 3. Python Interpreters and Environment Tooling

| Tool / Runtime | Location / Version | Status |
|---|---|---|
| **Base Python Interpreter** | `C:\Users\avkir\AppData\Local\Python\pythoncore-3.12-64\python.exe` | **Python 3.12.10** (64-bit AMD64, MSC v.1943) |
| **Windows App Execution Alias** | `C:\Users\avkir\AppData\Local\Microsoft\WindowsApps\python.exe` | Present |
| **Python Launcher (`py`)** | `py.exe` -> resolves solely to `pythoncore-3.12-64` (Python 3.12.10) | Verified via `py -0p` |
| **Base Package Manager** | `pip 25.0.1` | Built-in with Python 3.12 distribution |
| **Environment Managers** | Standard library `venv` available; `conda`, `uv`, `poetry` not present | Tested via `Get-Command` |
| **Shell** | Windows PowerShell 5.1 (Build `10.0.26100.9711`, Desktop Edition) | `$PSVersionTable` |

---

## 4. Network Reachability

| Target Endpoint | Port | Protocol | Test Result |
|---|---|---|---|
| `pypi.org` | 443 | HTTPS / TCP | **Reachable** (Connected to 151.101.0.223:443 via `Test-NetConnection`) |
| `files.pythonhosted.org` | 443 | HTTPS / TCP | **Reachable** (Connected to 151.101.0.223:443 via `Test-NetConnection`) |

> [!NOTE]
> Network reachability tests accessed only public official package endpoints. No credentials, tokens, or private endpoints were accessed, requested, or logged.

---

## 5. Source Tree and Transfer Provenance

- **Checkout Root:** `C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main`
- **Revision / Git Status:** Unversioned transfer checkout (no `.git` directory present).
- **Transfer Manifest Hash:** `ef9f538c971143c93773c45c929812a18247da78a10cec369f2205350d6e32cd`  
  *(Computed as SHA-256 over relative path and content hash of all 277 source files prior to environment setup)*.
- **Working Tree Integrity:** All 277 original source files preserved unmodified. No existing source files, data files, or archives were altered or deleted.

---

## 6. Compatibility Gate Assessment

| Criteria | Required | Observed on Host | Gate Decision |
|---|---|---|---|
| **OS Compatibility** | Windows 10/11 64-bit | Windows 11 Home 64-bit (Build 26200) | **PASS** |
| **Python Support** | Python 3.10 – 3.14 (per `qpiai-opt` & `qpiai-quantum`) | Python 3.12.10 64-bit AMD64 | **PASS** |
| **Binary Wheels** | Windows `cp312-win_amd64` wheels on PyPI | `qpiai-opt 0.0.9`, `ortools 9.15.6755`, pure-Python `qpiai-quantum 0.3.0` | **PASS** |
| **Available Disk** | > 10 GB free space | 511.63 GB free on `C:` | **PASS** |
| **Network Access** | PyPI access for wheel download | PyPI and PythonHosted reachable on 443 | **PASS** |

**Conclusion:** The host environment meets all compatibility gates. Isolated project-local environment creation proceeded safely under Stage B.
