"""Prepare CIFAR-10 from the course's Google Drive download.

This downloads the dataset *only if* torchvision's original CIFAR-10 batch files
are not already available with their official checksums. It supports ZIP and
TAR/TAR.GZ archives containing the standard `cifar-10-batches-py` directory.
"""
from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

DRIVE_FILE_ID = "1yUOoLlV7yFkKx-gAYJTFbw6YPbIdumjq"
DRIVE_LINK = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing"


def _cifar_filenames():
    from torchvision.datasets import CIFAR10
    return {
        *[name for name, _ in CIFAR10.train_list],
        *[name for name, _ in CIFAR10.test_list],
        CIFAR10.meta["filename"],
    }


def cifar10_is_ready(root: str | Path) -> bool:
    """Check the original torchvision CIFAR-10 batch-file checksums."""
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity

    root = Path(root) / CIFAR10.base_folder
    files = CIFAR10.train_list + CIFAR10.test_list + [
        (CIFAR10.meta["filename"], CIFAR10.meta["md5"])
    ]
    return all(check_integrity(str(root / filename), md5) for filename, md5 in files)


def _target_for_member(member_name: str, root: Path, permitted: set[str]) -> Path | None:
    """Extract only known CIFAR-10 files: no arbitrary archive paths are trusted."""
    bits = PurePosixPath(member_name.replace('\\', '/')).parts
    if 'cifar-10-batches-py' not in bits:
        return None
    i = bits.index('cifar-10-batches-py')
    if len(bits) != i + 2 or bits[-1] not in permitted:
        return None
    return root / 'cifar-10-batches-py' / bits[-1]


def unpack_cifar_archive(archive: str | Path, root: str | Path) -> int:
    """Unpack only standard CIFAR-10 batch files; return the count extracted."""
    archive, root = Path(archive), Path(root)
    allowed = _cifar_filenames()
    extracted = set()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if entry.is_dir():
                    continue
                dest = _target_for_member(entry.filename, root, allowed)
                if dest is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as src, dest.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                extracted.add(dest.name)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, mode='r:*') as source:
            for entry in source:
                if not entry.isfile():
                    continue
                dest = _target_for_member(entry.name, root, allowed)
                if dest is None:
                    continue
                reader = source.extractfile(entry)
                if reader is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with reader, dest.open('wb') as dst:
                    shutil.copyfileobj(reader, dst)
                extracted.add(dest.name)
    else:
        raise ValueError(
            'The Google Drive file is not a supported ZIP or TAR archive. '
            'Please check the file format or supply the standard CIFAR-10 '
            '`cifar-10-batches-py` folder.\nCourse dataset link: ' + DRIVE_LINK
        )
    return len(extracted)


def ensure_cifar10(root: str | Path = 'data/') -> Path:
    """Download from the supplied course Drive link and verify with torchvision.

    Requires `gdown` (included in requirements.txt). Drive file sharing must
    allow the student to access the file. Download happens only when needed.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if cifar10_is_ready(root):
        print('CIFAR-10 already present and verified; skipping download.')
        return root

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError('Install the course requirements first: python -m pip install -r requirements.txt') from exc

    destination = root / '_course_cifar10_download'
    print('Downloading CIFAR-10 from the course Google Drive link (first run only)...')
    print(DRIVE_LINK)
    try:
        result = gdown.download(id=DRIVE_FILE_ID, output=str(destination), quiet=False)
        if not result or not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError('The dataset could not be downloaded. Check Drive sharing permissions and network connectivity.')
        number = unpack_cifar_archive(destination, root)
        if number == 0:
            raise RuntimeError('The archive did not contain the expected `cifar-10-batches-py` files.')
        if not cifar10_is_ready(root):
            raise RuntimeError(
                'Dataset files were extracted but their checksums do not match the '
                'standard torchvision CIFAR-10 dataset. Confirm the Google Drive file '
                'is the original CIFAR-10 Python-format dataset archive.'
            )
    finally:
        if destination.is_file():
            destination.unlink()  # avoid leaving another ~170 MB copy
    print('CIFAR-10 ready and verified.')
    return root


if __name__ == '__main__':
    ensure_cifar10('data/')
