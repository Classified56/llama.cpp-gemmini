# Bundled Gemmini software interface

This directory contains the minimal Gemmini software dependency chain required
by the GGML Gemmini backend's `tiled_matmul_auto()` wrapper.

Bundled files:

- `include/gemmini.h`
- `include/gemmini_params.h`
- `include/gemmini_counter.h`
- `rocc-software/src/xcustom.h`

The relative directory layout is intentionally preserved so the upstream
headers do not need to be modified.

## Configuration compatibility

`include/gemmini_params.h` is generated from a particular Gemmini hardware
configuration. A binary can compile successfully with the wrong parameter
header and still execute incorrectly or issue incompatible commands.

The bundled header supplied with this starter describes an integer Gemmini
configuration with `DIM=16`, `elem_t=int8_t`, `acc_t=int32_t`, and
`XCUSTOM_ACC=3`.

For a different bitstream, configure with:

```bash
-DGGML_GEMMINI_PARAMS_FILE=/path/to/matching/gemmini_params.h
```

This copies the alternative generated header into a build-tree include overlay;
it does not modify the repository copy.

## Provenance

Before committing this subtree, record the exact source revision used to obtain
these files. The provided `scripts/update-gemmini-vendor.sh` does this
automatically when the source directory is a Git checkout.

Do not remove upstream copyright or license notices.

Gemmini source commit:
Chipyard source commit:
FireSim TARGET_CONFIG:
XCUSTOM_ACC:
DIM:
BANK_NUM:
BANK_ROWS:
ACC_ROWS:
elem_t:
acc_t:
gemmini_params.h SHA256: