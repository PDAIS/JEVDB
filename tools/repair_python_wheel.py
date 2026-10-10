"""Bundle Linux dependencies without losing DuckDB's appended extension metadata."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--plat", required=True)
    parser.add_argument("--openssl-license", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, default=Path("wheelhouse"))
    args = parser.parse_args()
    member = "jevdb/jevdb.duckdb_extension"
    with ZipFile(args.wheel) as archive:
        metadata = archive.read(member)[-534:]
    if not metadata.startswith(b"\x00\x93\x04\x10duckdb_signature\x80\x04"):
        raise ValueError("The input wheel has no valid DuckDB metadata trailer.")
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        subprocess.run([
            sys.executable, "-m", "auditwheel", "repair", "--plat", args.plat,
            "-w", str(root / "repaired"), str(args.wheel),
        ], check=True)
        repaired, = (root / "repaired").glob("*.whl")
        subprocess.run([sys.executable, "-m", "wheel", "unpack", str(repaired), "-d", str(root / "unpacked")], check=True)
        package, = (root / "unpacked").iterdir()
        licenses = package / "jevdb/licenses"
        licenses.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.openssl_license, licenses / "OpenSSL.txt")
        binary = package / member
        # patchelf moves ELF sections and may overwrite the appended DuckDB footer.
        if binary.read_bytes()[-534:] != metadata:
            with binary.open("ab") as stream:
                stream.write(metadata)
        # Repacking also updates the RECORD hashes after restoring the metadata.
        subprocess.run([sys.executable, "-m", "wheel", "pack", str(package), "-d", str(args.wheel_dir)], check=True)


if __name__ == "__main__":
    main()
