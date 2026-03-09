use super::traits::RuntimeAdapter;
use std::path::{Path, PathBuf};

/// Apply Windows-specific flags to hide console windows when spawning processes.
/// On non-Windows platforms this is a no-op.
#[cfg(target_os = "windows")]
pub(crate) fn hide_windows_console(cmd: &mut tokio::process::Command) {
    use std::os::windows::process::CommandExt;
    // CREATE_NO_WINDOW (0x0800_0000) prevents a visible console window.
    cmd.creation_flags(0x0800_0000);
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn hide_windows_console(_cmd: &mut tokio::process::Command) {}

/// Same helper for `std::process::Command`.
#[cfg(target_os = "windows")]
pub(crate) fn hide_windows_console_std(cmd: &mut std::process::Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000);
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn hide_windows_console_std(_cmd: &mut std::process::Command) {}

/// Native runtime — full access, runs on Mac/Linux/Docker/Raspberry Pi
pub struct NativeRuntime {
    shell: Option<ShellProgram>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShellProgram {
    kind: ShellKind,
    program: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShellKind {
    Sh,
    Bash,
    Pwsh,
    PowerShell,
    Cmd,
}

impl ShellKind {
    fn as_str(self) -> &'static str {
        match self {
            ShellKind::Sh => "sh",
            ShellKind::Bash => "bash",
            ShellKind::Pwsh => "pwsh",
            ShellKind::PowerShell => "powershell",
            ShellKind::Cmd => "cmd",
        }
    }
}

impl ShellProgram {
    fn add_shell_args(&self, process: &mut tokio::process::Command, command: &str) {
        match self.kind {
            ShellKind::Sh | ShellKind::Bash => {
                process.arg("-c").arg(command);
            }
            ShellKind::Pwsh | ShellKind::PowerShell => {
                process
                    .arg("-NoLogo")
                    .arg("-NoProfile")
                    .arg("-NonInteractive")
                    .arg("-Command")
                    .arg(command);
            }
            ShellKind::Cmd => {
                process.arg("/C").arg(command);
            }
        }
    }
}

fn detect_native_shell() -> Option<ShellProgram> {
    detect_native_shell_with_preference(None)
}

fn detect_native_shell_with_preference(preferred: Option<&str>) -> Option<ShellProgram> {
    #[cfg(target_os = "windows")]
    {
        let comspec = std::env::var_os("COMSPEC").map(PathBuf::from);
        detect_native_shell_with(true, |name| which::which(name).ok(), comspec, preferred)
    }
    #[cfg(not(target_os = "windows"))]
    {
        detect_native_shell_with(false, |name| which::which(name).ok(), None, preferred)
    }
}

fn detect_native_shell_with<F>(
    is_windows: bool,
    mut resolve: F,
    comspec: Option<PathBuf>,
    preferred: Option<&str>,
) -> Option<ShellProgram>
where
    F: FnMut(&str) -> Option<PathBuf>,
{
    // If user specified a preferred shell, try that first
    if let Some(pref) = preferred {
        let pref_lower = pref.trim().to_ascii_lowercase();
        // Map preference string to shell kind
        let preferred_kind = match pref_lower.as_str() {
            "powershell" | "ps" => Some(ShellKind::PowerShell),
            "pwsh" => Some(ShellKind::Pwsh),
            "bash" => Some(ShellKind::Bash),
            "sh" => Some(ShellKind::Sh),
            "cmd" | "cmd.exe" => Some(ShellKind::Cmd),
            "auto" | "" => None, // Fall through to auto-detection
            _ => None,
        };

        if let Some(kind) = preferred_kind {
            // Try to resolve the preferred shell
            let names: &[&str] = match kind {
                ShellKind::PowerShell => &["powershell", "powershell.exe"],
                ShellKind::Pwsh => &["pwsh", "pwsh.exe"],
                ShellKind::Bash => &["bash", "bash.exe"],
                ShellKind::Sh => &["sh", "sh.exe"],
                ShellKind::Cmd => &["cmd", "cmd.exe"],
            };
            for name in names {
                if let Some(program) = resolve(name) {
                    // Skip WSL bash launcher on Windows
                    if *name == "bash" && is_windows && is_windows_wsl_bash_launcher(&program) {
                        continue;
                    }
                    return Some(ShellProgram { kind, program });
                }
            }
            // If preferred shell not found on Windows, try COMSPEC for cmd
            if is_windows && kind == ShellKind::Cmd {
                if let Some(program) = comspec.clone() {
                    return Some(ShellProgram {
                        kind: ShellKind::Cmd,
                        program,
                    });
                }
            }
        }
    }

    // Fall back to auto-detection
    if is_windows {
        // On Windows, when no preference is set, prefer PowerShell for better Windows compatibility
        // This ensures LLM-generated commands use PowerShell syntax by default
        for (name, kind) in [
            ("pwsh", ShellKind::Pwsh),
            ("powershell", ShellKind::PowerShell),
            ("bash", ShellKind::Bash),
            ("sh", ShellKind::Sh),
            ("cmd", ShellKind::Cmd),
            ("cmd.exe", ShellKind::Cmd),
        ] {
            if let Some(program) = resolve(name) {
                // Windows may expose `C:\Windows\System32\bash.exe`, a legacy
                // WSL launcher that executes commands inside Linux userspace.
                // That breaks native Windows commands like `ipconfig`.
                if name == "bash" && is_windows_wsl_bash_launcher(&program) {
                    continue;
                }
                return Some(ShellProgram { kind, program });
            }
        }
        if let Some(program) = comspec {
            return Some(ShellProgram {
                kind: ShellKind::Cmd,
                program,
            });
        }
        return None;
    }

    for (name, kind) in [("sh", ShellKind::Sh), ("bash", ShellKind::Bash)] {
        if let Some(program) = resolve(name) {
            return Some(ShellProgram { kind, program });
        }
    }
    None
}

fn is_windows_wsl_bash_launcher(program: &Path) -> bool {
    let normalized = program
        .to_string_lossy()
        .replace('/', "\\")
        .to_ascii_lowercase();
    normalized.ends_with("\\windows\\system32\\bash.exe")
        || normalized.ends_with("\\windows\\sysnative\\bash.exe")
}

fn missing_shell_error() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "Native runtime could not find a usable shell (tried: bash, sh, pwsh, powershell, cmd). \
         Install Git Bash or PowerShell and ensure it is available on PATH."
    }
    #[cfg(not(target_os = "windows"))]
    {
        "Native runtime could not find a usable shell (tried: sh, bash). \
         Install a POSIX shell and ensure it is available on PATH."
    }
}

impl NativeRuntime {
    pub fn new() -> Self {
        Self {
            shell: detect_native_shell(),
        }
    }

    /// Create a NativeRuntime with a preferred shell.
    ///
    /// # Arguments
    /// * `preferred` - Preferred shell: "powershell", "pwsh", "bash", "sh", "cmd", or "auto"
    pub fn with_preferred_shell(preferred: Option<&str>) -> Self {
        Self {
            shell: detect_native_shell_with_preference(preferred),
        }
    }

    pub(crate) fn selected_shell_kind(&self) -> Option<&'static str> {
        self.shell.as_ref().map(|shell| shell.kind.as_str())
    }

    pub(crate) fn selected_shell_program(&self) -> Option<&Path> {
        self.shell.as_ref().map(|shell| shell.program.as_path())
    }

    /// Returns true if the selected shell uses PowerShell syntax.
    pub fn is_powershell_syntax(&self) -> bool {
        matches!(
            self.shell.as_ref().map(|s| s.kind),
            Some(ShellKind::Pwsh) | Some(ShellKind::PowerShell)
        )
    }

    /// Returns true if the selected shell uses Unix/bash syntax.
    pub fn is_unix_syntax(&self) -> bool {
        matches!(
            self.shell.as_ref().map(|s| s.kind),
            Some(ShellKind::Sh) | Some(ShellKind::Bash)
        )
    }

    /// Returns a description of the shell for use in system prompts.
    pub fn shell_prompt_hint(&self) -> &'static str {
        match self.shell.as_ref().map(|s| s.kind) {
            Some(ShellKind::Pwsh) | Some(ShellKind::PowerShell) => {
                "PowerShell (use PowerShell cmdlets and syntax, e.g., Get-ChildItem, Get-Content, Remove-Item)"
            }
            Some(ShellKind::Cmd) => {
                "Command Prompt (use cmd.exe syntax, e.g., dir, type, del)"
            }
            Some(ShellKind::Sh) | Some(ShellKind::Bash) => {
                "Unix shell (use standard Unix commands, e.g., ls, cat, rm)"
            }
            None => "Unknown shell",
        }
    }

    #[cfg(test)]
    fn new_for_test(shell: Option<ShellProgram>) -> Self {
        Self { shell }
    }
}

impl RuntimeAdapter for NativeRuntime {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "native"
    }

    fn has_shell_access(&self) -> bool {
        self.shell.is_some()
    }

    fn has_filesystem_access(&self) -> bool {
        true
    }

    fn storage_path(&self) -> PathBuf {
        directories::UserDirs::new().map_or_else(
            || PathBuf::from(".coraldesk"),
            |u| u.home_dir().join(".coraldesk"),
        )
    }

    fn supports_long_running(&self) -> bool {
        true
    }

    fn build_shell_command(
        &self,
        command: &str,
        workspace_dir: &Path,
    ) -> anyhow::Result<tokio::process::Command> {
        let shell = self
            .shell
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!(missing_shell_error()))?;

        let mut process = tokio::process::Command::new(&shell.program);
        shell.add_shell_args(&mut process, command);
        process.current_dir(workspace_dir);
        hide_windows_console(&mut process);
        Ok(process)
    }

    fn shell_prompt_hint(&self) -> &'static str {
        match self.shell.as_ref().map(|s| s.kind) {
            Some(ShellKind::Pwsh) | Some(ShellKind::PowerShell) => {
                "PowerShell (use PowerShell cmdlets and syntax, e.g., Get-ChildItem, Get-Content, Remove-Item)"
            }
            Some(ShellKind::Cmd) => {
                "Command Prompt (use cmd.exe syntax, e.g., dir, type, del)"
            }
            Some(ShellKind::Sh) | Some(ShellKind::Bash) => {
                "Unix shell (use standard Unix commands, e.g., ls, cat, rm)"
            }
            None => "Unknown shell",
        }
    }

    fn is_powershell_shell(&self) -> bool {
        matches!(
            self.shell.as_ref().map(|s| s.kind),
            Some(ShellKind::Pwsh) | Some(ShellKind::PowerShell)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn native_name() {
        assert_eq!(NativeRuntime::new().name(), "native");
    }

    #[test]
    fn native_has_shell_access() {
        assert_eq!(
            NativeRuntime::new().has_shell_access(),
            detect_native_shell().is_some()
        );
    }

    #[test]
    fn native_has_filesystem_access() {
        assert!(NativeRuntime::new().has_filesystem_access());
    }

    #[test]
    fn native_supports_long_running() {
        assert!(NativeRuntime::new().supports_long_running());
    }

    #[test]
    fn native_memory_budget_unlimited() {
        assert_eq!(NativeRuntime::new().memory_budget(), 0);
    }

    #[test]
    fn native_storage_path_contains_zeroclaw() {
        let path = NativeRuntime::new().storage_path();
        assert!(path.to_string_lossy().contains("zeroclaw"));
    }

    #[test]
    fn detect_shell_windows_prefers_powershell_by_default() {
        let mut map = HashMap::new();
        map.insert("bash", r"C:\Program Files\Git\bin\bash.exe");
        map.insert(
            "powershell",
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        );
        map.insert("cmd", r"C:\Windows\System32\cmd.exe");

        // Without preference, Windows now prefers PowerShell for better compatibility
        let shell = detect_native_shell_with(
            true,
            |name| map.get(name).map(PathBuf::from),
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
            None,
        )
        .expect("windows shell should be detected");

        assert_eq!(shell.kind, ShellKind::PowerShell);
    }

    #[test]
    fn detect_shell_windows_respects_bash_preference() {
        let mut map = HashMap::new();
        map.insert("bash", r"C:\Program Files\Git\bin\bash.exe");
        map.insert(
            "powershell",
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        );
        map.insert("cmd", r"C:\Windows\System32\cmd.exe");

        // With bash preference, should use bash
        let shell = detect_native_shell_with(
            true,
            |name| map.get(name).map(PathBuf::from),
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
            Some("bash"),
        )
        .expect("windows shell should be detected");

        assert_eq!(shell.kind, ShellKind::Bash);
    }

    #[test]
    fn detect_shell_windows_falls_back_to_powershell_then_cmd() {
        let mut map = HashMap::new();
        map.insert(
            "powershell",
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        );

        let shell = detect_native_shell_with(
            true,
            |name| map.get(name).map(PathBuf::from),
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
            None,
        )
        .expect("windows shell should be detected");

        assert_eq!(shell.kind, ShellKind::PowerShell);

        let cmd_shell = detect_native_shell_with(
            true,
            |_name| None,
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
            None,
        )
        .expect("cmd fallback should be detected");
        assert_eq!(cmd_shell.kind, ShellKind::Cmd);
    }

    #[test]
    fn detect_shell_windows_skips_system32_bash_wsl_launcher() {
        let mut map = HashMap::new();
        map.insert("bash", r"C:\Windows\System32\bash.exe");
        map.insert(
            "powershell",
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        );
        map.insert("cmd", r"C:\Windows\System32\cmd.exe");

        let shell = detect_native_shell_with(
            true,
            |name| map.get(name).map(PathBuf::from),
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
            None,
        )
        .expect("windows shell should be detected");

        assert_eq!(shell.kind, ShellKind::PowerShell);
        assert_eq!(
            shell.program,
            PathBuf::from(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
        );
    }

    #[test]
    fn detect_shell_windows_uses_cmd_when_only_wsl_bash_exists() {
        let mut map = HashMap::new();
        map.insert("bash", r"C:\Windows\Sysnative\bash.exe");

        let shell = detect_native_shell_with(
            true,
            |name| map.get(name).map(PathBuf::from),
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
            None,
        )
        .expect("cmd fallback should be detected");

        assert_eq!(shell.kind, ShellKind::Cmd);
        assert_eq!(shell.program, PathBuf::from(r"C:\Windows\System32\cmd.exe"));
    }

    #[test]
    fn wsl_launcher_detection_matches_known_paths() {
        assert!(is_windows_wsl_bash_launcher(Path::new(
            r"C:\Windows\System32\bash.exe"
        )));
        assert!(is_windows_wsl_bash_launcher(Path::new(
            r"C:\Windows\Sysnative\bash.exe"
        )));
        assert!(!is_windows_wsl_bash_launcher(Path::new(
            r"C:\Program Files\Git\bin\bash.exe"
        )));
    }

    #[test]
    fn detect_shell_unix_prefers_sh() {
        let mut map = HashMap::new();
        map.insert("sh", "/bin/sh");
        map.insert("bash", "/usr/bin/bash");

        let shell =
            detect_native_shell_with(false, |name| map.get(name).map(PathBuf::from), None, None)
                .expect("unix shell should be detected");

        assert_eq!(shell.kind, ShellKind::Sh);
    }

    #[test]
    fn native_without_shell_disables_shell_access() {
        let runtime = NativeRuntime::new_for_test(None);
        assert!(!runtime.has_shell_access());

        let err = runtime
            .build_shell_command("echo hello", Path::new("."))
            .expect_err("build should fail without available shell")
            .to_string();
        assert!(err.contains("could not find a usable shell"));
    }

    #[test]
    fn native_builds_powershell_command() {
        let runtime = NativeRuntime::new_for_test(Some(ShellProgram {
            kind: ShellKind::PowerShell,
            program: PathBuf::from("powershell"),
        }));

        let command = runtime
            .build_shell_command("Get-Location", Path::new("."))
            .expect("powershell command should build");
        let debug = format!("{command:?}");

        assert!(debug.contains("powershell"));
        assert!(debug.contains("-NoProfile"));
        assert!(debug.contains("-Command"));
        assert!(debug.contains("Get-Location"));
    }

    #[test]
    fn native_builds_cmd_command() {
        let runtime = NativeRuntime::new_for_test(Some(ShellProgram {
            kind: ShellKind::Cmd,
            program: PathBuf::from("cmd"),
        }));

        let command = runtime
            .build_shell_command("echo hello", Path::new("."))
            .expect("cmd command should build");
        let debug = format!("{command:?}");

        assert!(debug.contains("cmd"));
        assert!(debug.contains("/C"));
        assert!(debug.contains("echo hello"));
    }

    #[test]
    fn native_builds_shell_command() {
        let runtime = NativeRuntime::new();
        if !runtime.has_shell_access() {
            return;
        }

        let cwd = std::env::temp_dir();
        let command = runtime.build_shell_command("echo hello", &cwd).unwrap();
        let debug = format!("{command:?}");
        assert!(debug.contains("echo hello"));
    }
}
