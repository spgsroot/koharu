//! System headers for MSVC targets.
//!
//! libclang trusts `INCLUDE` whenever it is set, so a terminal that carries a
//! partial Visual Studio environment (toolchain headers without the Windows
//! SDK) hides `string.h` and fails every binding. MSVC builds therefore name the
//! toolchain and SDK header directories explicitly, found the same way the
//! Visual Studio installer and the SDK register them. When discovery fails the
//! list is empty and clang's own search applies unchanged.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) fn target_args() -> Vec<String> {
    let msvc =
        env::var("CARGO_CFG_TARGET_ENV").map_or(cfg!(target_env = "msvc"), |value| value == "msvc");
    if !msvc {
        return Vec::new();
    }
    let mut args = vec!["-fms-extensions".to_owned()];
    if let Ok(target) = env::var("TARGET") {
        args.push(format!("--target={target}"));
    }
    args
}

pub(crate) fn system_include_dirs() -> Vec<PathBuf> {
    let msvc =
        env::var("CARGO_CFG_TARGET_ENV").map_or(cfg!(target_env = "msvc"), |value| value == "msvc");
    if !msvc {
        return Vec::new();
    }
    let mut dirs = Vec::new();
    if let Some(toolchain) = toolchain_include() {
        dirs.push(toolchain);
    }
    if let Some(sdk) = sdk_include() {
        dirs.extend(["ucrt", "um", "shared"].map(|part| sdk.join(part)));
    }
    dirs.retain(|dir| dir.is_dir());
    dirs
}

fn toolchain_include() -> Option<PathBuf> {
    let installer = PathBuf::from(env::var_os("ProgramFiles(x86)")?)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let output = Command::new(installer)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    let installation = String::from_utf8(output.stdout).ok()?;
    let tools = Path::new(installation.trim())
        .join("VC")
        .join("Tools")
        .join("MSVC");
    newest_child(&tools, |version| {
        version.join("include").join("vcruntime.h").is_file()
    })
    .map(|version| version.join("include"))
}

fn sdk_include() -> Option<PathBuf> {
    let output = Command::new("reg")
        .args([
            "query",
            r"HKLM\SOFTWARE\Microsoft\Windows Kits\Installed Roots",
            "/v",
            "KitsRoot10",
        ])
        .output()
        .ok()?;
    let listing = String::from_utf8(output.stdout).ok()?;
    let root = listing
        .lines()
        .find_map(|line| line.trim().strip_prefix("KitsRoot10"))?
        .trim()
        .strip_prefix("REG_SZ")?
        .trim();
    newest_child(&Path::new(root).join("Include"), |version| {
        version.join("ucrt").join("string.h").is_file()
    })
}

/// The highest dotted-version subdirectory of `parent` that passes `usable`.
fn newest_child(parent: &Path, usable: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let mut versions = std::fs::read_dir(parent)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| usable(path))
        .filter_map(|path| {
            let key = path
                .file_name()?
                .to_str()?
                .split('.')
                .map(|part| part.parse::<u32>().ok())
                .collect::<Option<Vec<_>>>()?;
            Some((key, path))
        })
        .collect::<Vec<_>>();
    versions.sort();
    versions.pop().map(|(_, path)| path)
}
