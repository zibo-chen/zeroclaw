use std::path::Path;

/// Returns `true` when a file has more than one hard link.
///
/// Multiple hard links allow path-based workspace guards to be bypassed by
/// linking a workspace path to external sensitive content.
pub fn has_multiple_hard_links(path: &Path) -> bool {
    link_count(path) > 1
}

#[cfg(unix)]
fn link_count(path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).map(|m| m.nlink()).unwrap_or(1)
}

#[cfg(windows)]
fn link_count(path: &Path) -> u64 {
    // `MetadataExt::number_of_links()` requires the unstable `windows_by_handle`
    // feature gate, and Win32 FFI would require unsafe code which is forbidden in
    // this crate. Use `fsutil hardlink list` instead — it is available on all NTFS
    // volumes without administrator privileges for normal user files, and returns
    // one path per line (including the file itself), so the line count equals the
    // hard-link count.
    use std::process::Command;

    let output = match Command::new("fsutil")
        .args(["hardlink", "list"])
        .arg(path)
        .output()
    {
        Ok(out) => out,
        Err(_) => return 1,
    };

    if !output.status.success() {
        // Non-NTFS volumes (FAT32, exFAT) do not support hard links at all,
        // so failing here is safe to treat as "no multiple links".
        return 1;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let count = text.lines().filter(|l| !l.trim().is_empty()).count();
    count.max(1) as u64
}

#[cfg(not(any(unix, windows)))]
fn link_count(_path: &Path) -> u64 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_link_file_is_not_flagged() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("single.txt");
        std::fs::write(&file, "hello").unwrap();
        assert!(!has_multiple_hard_links(&file));
    }

    #[test]
    fn hard_link_file_is_flagged_when_supported() {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("original.txt");
        let linked = dir.path().join("linked.txt");
        std::fs::write(&original, "hello").unwrap();

        if std::fs::hard_link(&original, &linked).is_err() {
            // Some filesystems may disable hard links; treat as unsupported.
            return;
        }

        assert!(has_multiple_hard_links(&original));
    }
}
