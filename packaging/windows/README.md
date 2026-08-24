# radiarandom on Windows

## The short version

**Windows has no supported way to add entropy to the operating system's RNG.**
There is no `rngd` equivalent, and this project does not pretend otherwise. What
it offers instead is a service that *hands* entropy to applications that ask
for it.

## Why there is no Windows equivalent of `radiarandom feed`

Linux exposes `RNDADDENTROPY` on `/dev/random`: a process with `CAP_SYS_ADMIN`
can mix a buffer into the kernel pool **and** credit a specific number of
entropy bits, which then benefits every consumer of `getrandom(2)`. Windows has
no counterpart:

- `BCryptGenRandom` accepts `BCRYPT_RNG_USE_ENTROPY_IN_BUFFER` (`0x00000001`),
  which historically mixed the caller's buffer into the result. Microsoft's
  reference for the function states, verbatim: *"Windows 8 and later: This flag
  is ignored in Windows 8 and later."* On any supported Windows it does nothing.
  <https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom>
- `CryptGenRandom`, the legacy CryptoAPI call that did mix caller-supplied data,
  is deprecated and now sits on top of the same CNG DRBG.
- `HKLM\SOFTWARE\Microsoft\Cryptography\RNG\Seed` is a legacy artefact and is
  not an entropy input for modern CNG.
- The kernel's entropy gathering lives in `cng.sys` and exposes no user-mode
  contribution interface.

Any tool claiming to "add entropy to Windows" on a current build is either
mistaken or is doing what this one does: serving entropy to consumers that opt
in.

## What you get instead

### 1. A named-pipe entropy service

```powershell
radiarandom serve --transport pipe
```

Serves on `\\.\pipe\radiarandom`. The pipe is created with
`PIPE_REJECT_REMOTE_CLIENTS`, so it is reachable only from the local machine,
and with the default security descriptor, which grants access to the creating
user and to administrators.

Read it from PowerShell:

```powershell
$pipe = New-Object System.IO.Pipes.NamedPipeClientStream(
    '.', 'radiarandom', [System.IO.Pipes.PipeDirection]::In)
$pipe.Connect(5000)
$buffer = New-Object byte[] 32
$pipe.Read($buffer, 0, 32) | Out-Null
[BitConverter]::ToString($buffer) -replace '-'
$pipe.Dispose()
```

Or from Python:

```python
with open(r'\\.\pipe\radiarandom', 'rb') as pipe:
    key = pipe.read(32)
```

### 2. A loopback TCP service

```powershell
radiarandom serve --transport tcp --port 7373
```

Convenient for containers and VMs. It refuses to bind anything other than
loopback unless you pass `--allow-remote`, because streaming key material over
a network in the clear is almost never what you want.

### 3. Seed files and direct generation

```powershell
radiarandom seed-file C:\ProgramData\radiarandom\seed.bin -n 64
radiarandom gen -n 32 --format hex
radiarandom gen -n 1048576 --format bin -o random.bin
```

## Running it as a Windows service

There is no built-in service wrapper; use a supervisor. With
[NSSM](https://nssm.cc/):

```powershell
# Run these from an elevated prompt.
nssm install radiarandom "C:\path\to\.venv\Scripts\radiarandom.exe" "serve --transport pipe --quiet"
nssm set radiarandom AppDirectory "C:\path\to\radiarandom"
nssm set radiarandom Description "RadiaCode hardware entropy service"
nssm set radiarandom Start SERVICE_AUTO_START
# The start-up test needs several minutes on background radiation.
nssm set radiarandom AppThrottle 60000
nssm set radiarandom AppStdout "C:\ProgramData\radiarandom\service.log"
nssm set radiarandom AppStderr "C:\ProgramData\radiarandom\service.log"
nssm start radiarandom
```

Or with the built-in `sc.exe` plus a wrapper such as WinSW. Either way, run the
service as a dedicated low-privilege account that has access to the USB device,
and remember that the pipe's default ACL follows the account that creates it —
if the service runs as `LocalSystem`, ordinary users will not be able to read
the pipe without an explicit security descriptor.

## USB driver

The RadiaCode enumerates as `USB\VID_0483&PID_F123`. PyUSB needs it bound to a
libusb-compatible driver (WinUSB or libusbK); the vendor software installs
libusbK, which works. If `radiarandom info` reports no device, use
[Zadig](https://zadig.akeo.ie/) to bind WinUSB to the interface.

## A Windows-specific gotcha

PyUSB will happily load whatever `libusb-1.0.dll` it finds first on `PATH`, and
a mismatched one crashes the interpreter with an access violation inside the
first control transfer — a hard segfault with no Python traceback. Install the
`usb` extra so the matched DLL is present, which `radiarandom` pins at import
time:

```powershell
pip install "radiarandom[usb]"
```

This was not hypothetical: it is exactly what happened on the development
machine, and `radiarandom._usbshim` exists to prevent it.
