//! The runner **admin surface** - the local control channel between a runner
//! and the manager.
//!
//! Transport is HTTP/1 over a **local-only** endpoint (the Docker precedent):
//! a Windows named pipe (`\\.\pipe\paddock-runner-<port>`, DACL = owning user
//! + SYSTEM only, `PIPE_REJECT_REMOTE_CLIENTS`) or a Unix domain socket
//!   (`$XDG_RUNTIME_DIR/paddock/runner-<port>.sock`, dir mode 0700). The OS *is*
//!   the authentication: same user => full admin, anyone else => can't connect.
//!   **This surface never binds TCP**, and inference API keys never grant admin
//!   ops - the separation is by transport, not policy.
//!
//! The wire contract is versioned (`WIRE_VERSION`) with a **v1-frozen core**:
//! identify + health + drain/shutdown never change shape, so any future
//! manager can recognize and cleanly stop any runner ever shipped. Everything
//! richer (stats, events) is capability-discovered via `identify`.

pub mod client;
pub mod codec;
pub mod metal_budget;
pub mod residency;
pub mod server;
pub mod types;
pub mod version;
#[cfg(windows)]
mod winsec;

use std::path::PathBuf;

/// The data root + which rung of the ladder chose it. One resolver for the
/// three distribution modes; manager and runner both link this crate, so the
/// two halves can never disagree about where data lives:
///
///   1. `PADDOCK_DATA` env - explicit override, always wins
///   2. `~/paddock` for a DEV build - an exe under a cargo `target/` keeps the
///      checkout's own models rather than growing a data root inside `target/`
///      that `cargo clean` would eat
///   3. `data\` beside the exe - PORTABLE, and CREATED if absent: a copy of the
///      folder is the whole world, wherever it lands
///   4. the machine root an installer created - `%ProgramData%\Paddock` on
///      Windows, `/var/lib/paddock` elsewhere - the per-box appliance mode
///   5. `~/paddock`, then the cwd - last resorts for an exe somewhere it may
///      not write
///
/// Rung 3 used to require `data\` to already exist, and nothing ever created
/// it but the packaging script. So a portable folder copied without its data
/// subtree - a drag-and-drop that skipped the multi-GB part, a zip tool that
/// dropped an empty dir, a copy taken while SQLite held the -wal open - landed
/// and quietly adopted `%USERPROFILE%\paddock`: on a machine with an
/// existing install, another install's models, servers and database. Exactly
/// what portable mode exists to prevent (found by copying a portable folder
/// somewhere else and reading the startup banner).
///
/// Portable now means what it says: unless you point somewhere else, the
/// program's own folder is where its data lives. Writability is settled by
/// TRYING to create the directory rather than by sniffing the path - a
/// read-only Program Files or a mounted image fails and falls through on its
/// own, with no list of special locations to keep current.
///
/// The source string feeds the startup banner: where data lives must be
/// STATED, not divined from behavior.
pub fn data_root_resolved() -> (PathBuf, &'static str) {
    if let Some(p) = std::env::var_os("PADDOCK_DATA").filter(|p| !p.is_empty()) {
        return (PathBuf::from(p), "PADDOCK_DATA");
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(PathBuf::from));
    // A signed .app is immutable, never a portable data directory. Keep an
    // existing user library in place; new app installations use Application
    // Support. Bundled runners resolve identically, including Helpers/ nesting.
    #[cfg(target_os = "macos")]
    if let (Some(exe), Some(user_home)) = (&exe_dir, std::env::var_os("HOME"))
        && let Some(root) = macos_app_data_root(exe, std::path::Path::new(&user_home))
    {
        return (root, "macOS application");
    }
    if let Some(dir) = exe_dir.as_ref().filter(|d| !in_cargo_target(d)) {
        let portable = dir.join("data");
        // is_dir first so the common case costs one stat, then create for the
        // first run of a fresh copy. `create_dir_all` is idempotent, so the
        // race with a second process starting beside us is a non-event.
        if portable.is_dir() || std::fs::create_dir_all(&portable).is_ok() {
            return (portable, "portable");
        }
    }
    let machine = if cfg!(windows) {
        std::env::var_os("ProgramData").map(|p| PathBuf::from(p).join("Paddock"))
    } else {
        Some(PathBuf::from("/var/lib/paddock"))
    };
    if let Some(m) = machine
        && m.is_dir()
    {
        return (m, "installed");
    }
    match std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        Some(home) => (PathBuf::from(home).join("paddock"), "home"),
        None => (PathBuf::from("."), "cwd"),
    }
}

#[cfg(any(target_os = "macos", test))]
fn macos_app_data_root(exe_dir: &std::path::Path, user_home: &std::path::Path) -> Option<PathBuf> {
    let is_app = exe_dir.ancestors().any(|dir| {
        dir.file_name().is_some_and(|name| name == "Contents")
            && dir
                .parent()
                .and_then(|p| p.extension())
                .is_some_and(|ext| ext == "app")
    });
    if !is_app {
        return None;
    }
    let existing = user_home.join("paddock");
    let application = user_home.join("Library/Application Support/Paddock");
    // Once this app has established a library, a later CLI invocation creating
    // ~/paddock must not silently switch it to a different database on restart.
    Some(if application.join("paddock.db").is_file() {
        application
    } else if existing.is_dir() {
        existing
    } else {
        application
    })
}

/// Is this exe a cargo build artifact - `target/{debug,release}/paddock.exe`?
///
/// Asked so a `cargo run` does not become "portable" and start a fresh, empty
/// data root inside `target/`, orphaning the checkout's models and losing the
/// lot to the next `cargo clean`.
///
/// The test is cargo's own marker, not a path-name guess: cargo writes a
/// `CACHEDIR.TAG` at the root of every target dir (the freedesktop
/// cache-directory convention, so backup tools skip it). A user directory that
/// happens to be called "release" has no such file; a target dir renamed by
/// `CARGO_TARGET_DIR` still does.
fn in_cargo_target(exe_dir: &std::path::Path) -> bool {
    exe_dir
        .parent()
        .is_some_and(|p| p.join("CACHEDIR.TAG").is_file())
}

/// [`data_root_resolved`] without the provenance tag.
pub fn data_root() -> PathBuf {
    data_root_resolved().0
}

/// The pipe name (Windows) for a runner's admin surface, keyed by its
/// inference port - one runner per port, one pipe per runner.
#[cfg(windows)]
pub fn pipe_name(port: u16) -> String {
    format!(r"\\.\pipe\paddock-runner-{port}")
}

/// Per-boot runtime dir for admin sockets (Unix): `$XDG_RUNTIME_DIR/paddock/`,
/// falling back to `~/paddock/runtime/` (created 0700).
#[cfg(unix)]
pub fn runtime_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(x).join("paddock");
    }
    match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h).join("paddock").join("runtime"),
        None => PathBuf::from("./paddock-runtime"),
    }
}

/// The socket path (Unix) for a runner's admin surface.
#[cfg(unix)]
pub fn socket_path(port: u16) -> PathBuf {
    runtime_dir().join(format!("runner-{port}.sock"))
}

/// Where this port's admin endpoint lives, spelled for a person.
///
/// For refusals: "something is using this port" is only actionable if the
/// operator can find the something. A DGX Spark user lost an evening to a
/// leftover socket precisely because every message about it was silent on
/// where it was.
#[cfg(unix)]
pub fn endpoint_display(port: u16) -> String {
    socket_path(port).display().to_string()
}

/// Where this port's admin endpoint lives, spelled for a person. Windows pipes
/// vanish with their process, so this is for symmetry of message, not rescue.
#[cfg(windows)]
pub fn endpoint_display(port: u16) -> String {
    pipe_name(port)
}

/// Enumerate ports with an admin endpoint on this host - the manager's
/// reconciliation input, and what it asks before calling a port taken.
///
/// Windows pipes disappear with their process, so presence there is close to
/// liveness. A Unix socket file does not: it stays after its runner dies, and
/// counting those files made a stopped endpoint look like it was still
/// serving - the manager then refused to start it again. So on Unix a file
/// only counts when something is listening on it (`has_listener`). That is
/// still not a health check: a hung runner keeps its listener, and
/// `identify` remains the liveness question for anything that needs one.
pub fn enumerate() -> Vec<u16> {
    #[cfg(unix)]
    {
        enumerate_dir(&runtime_dir())
    }
    #[cfg(not(unix))]
    {
        let mut ports = Vec::new();
        // The pipe namespace is enumerable as a directory listing.
        #[cfg(windows)]
        if let Ok(entries) = std::fs::read_dir(r"\\.\pipe\") {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(p) = name.strip_prefix("paddock-runner-")
                    && let Ok(port) = p.parse::<u16>()
                {
                    ports.push(port);
                }
            }
        }
        ports.sort_unstable();
        ports
    }
}

/// The Unix half of `enumerate`, over any directory (tests point it at a
/// temp dir rather than the real runtime dir).
#[cfg(unix)]
fn enumerate_dir(dir: &std::path::Path) -> Vec<u16> {
    let mut ports = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(rest) = name.strip_prefix("runner-")
                && let Some(p) = rest.strip_suffix(".sock")
                && let Ok(port) = p.parse::<u16>()
                && has_listener(&e.path())
            {
                ports.push(port);
            }
        }
    }
    ports.sort_unstable();
    ports
}

/// Ask the kernel which unix sockets are actually listening, by path.
///
/// `/proc/net/unix` lists every bound unix socket in this network namespace as
/// `Num RefCount Protocol Flags Type St Inode Path`, and a listening socket
/// carries **SO_ACCEPTCON (0x10000)** in Flags. So the question "is anything
/// listening on this path" is answerable by reading a file, with no connect()
/// and therefore **no permission to the socket required** - which is the whole
/// point. `connect()` alone cannot tell a live runner from a socket we merely
/// may not open, and it used to answer "occupied" for both (see `has_listener`).
///
/// A corpse - a socket file whose runner is gone - does not appear here at all,
/// because the file on disk and the kernel object are different things: unlink
/// is what removes the file, close is what removes this entry. Verified on the
/// GB10: a bound+listening socket shows one row with Flags `00010000`, a
/// leftover file shows zero rows.
///
/// `None` means the question could not be asked (no procfs, unreadable, a path
/// that is not valid UTF-8) and the caller should fall back.
///
/// This re-reads the file once per socket rather than building a set for the
/// whole directory. That is deliberate: there is one caller, it looks at a
/// handful of runner sockets, and the file is a few hundred lines - so the
/// scan costs microseconds, and keeping it a pure path -> bool question is
/// worth more than the saved reads on a path this load-bearing.
#[cfg(target_os = "linux")]
fn listening_per_procfs(path: &std::path::Path) -> Option<bool> {
    const SO_ACCEPTCON: u32 = 0x0001_0000;
    let want = path.to_str()?;
    let txt = std::fs::read_to_string("/proc/net/unix").ok()?;
    for line in txt.lines().skip(1) {
        // Match the path by SUFFIX rather than by splitting into columns: the
        // path is the last field and may itself contain spaces, so taking
        // column 7 of a whitespace split would truncate such a path and miss.
        // The preceding whitespace check stops "/a/b.sock" matching a line
        // whose path is "/other/a/b.sock". An unbound socket has no path
        // column at all and simply never matches.
        let Some(head) = line.strip_suffix(want) else {
            continue;
        };
        if !head.ends_with(char::is_whitespace) {
            continue;
        }
        // Flags is the 4th column, hex, and everything before the path is
        // space-free - so a plain whitespace split is safe for it.
        if head
            .split_whitespace()
            .nth(3)
            .and_then(|f| u32::from_str_radix(f, 16).ok())
            .is_some_and(|v| v & SO_ACCEPTCON != 0)
        {
            return Some(true);
        }
    }
    // Present but not listening, or absent entirely: either way nothing will
    // accept on it.
    Some(false)
}

/// Is anything listening on this socket file?
///
/// On Linux the kernel is asked directly (`listening_per_procfs`), because the
/// probe below cannot distinguish "a live runner" from "a socket this process
/// may not open". That mattered: a socket left by a runner once started under
/// `sudo` is root-owned, `connect()` returns EACCES to the manager running as
/// the user, the port read as occupied FOREVER, and the operator could neither
/// start the endpoint nor remove its config (both gate on `enumerate`) without
/// knowing to go delete a file nothing told them about.
///
/// Elsewhere, a non-blocking connect answers without talking to the runner:
/// ECONNREFUSED means the file is a corpse, ENOENT that it went away while we
/// looked. Everything else keeps the port - a connection (the listener accepts
/// and sees us hang up, which it logs at trace), EAGAIN (a full backlog, so a
/// listener certainly exists), EACCES (a socket we may not open is still
/// somebody's, and without procfs we cannot do better than assume so).
///
/// Corpses are skipped, never deleted here: a runner between its bind() and
/// listen() refuses too, and its own bind already clears a stale file for its
/// port.
#[cfg(unix)]
fn has_listener(path: &std::path::Path) -> bool {
    #[cfg(target_os = "linux")]
    if let Some(listening) = listening_per_procfs(path) {
        return listening;
    }
    use socket2::{Domain, SockAddr, Socket, Type};
    use std::io::ErrorKind;
    let Ok(addr) = SockAddr::unix(path) else {
        return true;
    };
    let Ok(sock) = Socket::new(Domain::UNIX, Type::STREAM, None) else {
        return true;
    };
    if sock.set_nonblocking(true).is_err() {
        return true;
    }
    match sock.connect(&addr) {
        Ok(()) => true,
        Err(e) => !matches!(e.kind(), ErrorKind::ConnectionRefused | ErrorKind::NotFound),
    }
}

/// Default `RUST_LOG` when the environment does not set one.
///
/// Ours at `debug`, the plumbing at `warn`. The plumbing part is the point:
/// `rmcp` logs at INFO what an SDK reasonably logs - service lifecycle, task
/// cancellation, and the full JSON-RPC response body of every call. Embedded in
/// a host that talks to several MCP servers, that buries our own lines under
/// hundreds of characters of protocol per tool call, and a log nobody can read
/// is a log nobody reads. Their default is fine for a standalone client; it is
/// wrong for us, and the level is ours to choose.
///
/// `hyper`/`h2`/`tower_http` get the same treatment for the same reason -
/// per-connection and per-frame chatter that says nothing an operator acts on.
/// Anything genuinely wrong still arrives, because they are capped at warn, not
/// silenced.
///
/// One constant because there were three copies of this string (manager main,
/// runner startup, runner service) and they had already begun to matter
/// separately. `RUST_LOG` still overrides everything.
pub mod logging;

pub const DEFAULT_LOG_FILTER: &str =
    "info,paddock=debug,rmcp=warn,hyper=warn,h2=warn,tower_http=warn";

#[cfg(all(test, unix))]
mod enumerate_tests {
    use super::enumerate_dir;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("pd-enum-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("temp dir for the enumerate test");
        d
    }

    /// The DGX Spark failure, reduced: a runner's socket file left behind by
    /// a stopped runner must not read as a serving port, while a live one
    /// still does.
    #[test]
    fn a_socket_nobody_listens_on_is_not_a_serving_port() {
        let dir = tmp("corpse");
        let live = std::os::unix::net::UnixListener::bind(dir.join("runner-11540.sock")).unwrap();
        // bound then dropped: the file stays, the listener is gone
        drop(std::os::unix::net::UnixListener::bind(dir.join("runner-11541.sock")).unwrap());
        assert!(dir.join("runner-11541.sock").exists());
        // names that are not runner sockets never count
        std::fs::write(dir.join("runner-11542.sock.bak"), b"").unwrap();
        std::fs::write(dir.join("notes.txt"), b"").unwrap();

        assert_eq!(enumerate_dir(&dir), vec![11540]);
        // skipped, not deleted - the runner that next binds 11541 clears it
        assert!(dir.join("runner-11541.sock").exists());

        drop(live);
        assert_eq!(enumerate_dir(&dir), Vec::<u16>::new());
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The procfs probe is the one that works on a socket we may not connect to,
/// so it has to be right about paths as well as about liveness.
#[cfg(all(test, target_os = "linux"))]
mod procfs_listener_tests {
    use super::listening_per_procfs;
    use std::os::unix::net::UnixListener;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("pd-procfs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("temp dir for the procfs test");
        d
    }

    /// The distinction the whole fix rests on: the file on disk and the kernel
    /// object are different things, and only the kernel knows which is which.
    #[test]
    fn a_listener_is_visible_and_a_corpse_file_is_not() {
        let dir = tmp("live");
        let path = dir.join("runner-11540.sock");
        let live = UnixListener::bind(&path).unwrap();
        assert_eq!(listening_per_procfs(&path), Some(true));

        // Closing the listener leaves the file exactly where it was.
        drop(live);
        assert!(path.exists(), "the corpse file outlives its listener");
        assert_eq!(listening_per_procfs(&path), Some(false));

        // A path nothing ever bound.
        assert_eq!(
            listening_per_procfs(&dir.join("runner-9999.sock")),
            Some(false)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The path is the last column and may contain spaces, so it is matched by
    /// suffix rather than by splitting into columns - a column split would
    /// truncate this path and report the live runner as a corpse.
    #[test]
    fn a_socket_path_containing_spaces_still_matches() {
        let dir = tmp("spaced").join("a dir with spaces");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("runner-11540.sock");
        let live = UnixListener::bind(&path).unwrap();
        assert_eq!(listening_per_procfs(&path), Some(true));
        drop(live);
        std::fs::remove_dir_all(dir.parent().unwrap()).ok();
    }

    /// Suffix matching must not make one socket answer for another: a live
    /// `/<tmp>/deep/runner-11540.sock` says nothing about `/runner-11540.sock`.
    #[test]
    fn a_suffix_of_a_live_path_is_not_that_path() {
        let dir = tmp("suffix");
        let deep = dir.join("deep");
        std::fs::create_dir_all(&deep).unwrap();
        let live = UnixListener::bind(deep.join("runner-11540.sock")).unwrap();
        assert_eq!(
            listening_per_procfs(std::path::Path::new("/runner-11540.sock")),
            Some(false),
            "a bare suffix must not inherit a real socket's liveness"
        );
        drop(live);
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod data_root_tests {
    use super::in_cargo_target;

    #[test]
    fn macos_bundle_never_becomes_portable_data() {
        let dir = std::env::temp_dir().join(format!("pd-app-root-{}", std::process::id()));
        let root = super::macos_app_data_root(
            std::path::Path::new("/Applications/Paddock.app/Contents/MacOS"),
            &dir,
        );
        assert_eq!(root, Some(dir.join("Library/Application Support/Paddock")));
        std::fs::create_dir_all(dir.join("paddock")).unwrap();
        assert_eq!(
            super::macos_app_data_root(
                std::path::Path::new("/Applications/Paddock.app/Contents/Helpers/runner"),
                &dir
            ),
            Some(dir.join("paddock"))
        );
        assert!(super::macos_app_data_root(std::path::Path::new("/usr/local/bin"), &dir).is_none());
        let application = dir.join("Library/Application Support/Paddock");
        std::fs::create_dir_all(&application).unwrap();
        std::fs::write(application.join("paddock.db"), b"test marker, not a DB").unwrap();
        assert_eq!(
            super::macos_app_data_root(
                std::path::Path::new("/Applications/Paddock.app/Contents/MacOS"),
                &dir
            ),
            Some(application)
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The dev carve-out keys on cargo's own CACHEDIR.TAG, so a directory that
    /// merely LOOKS like a build output is not one. This is the difference
    /// between "portable" and "a data root cargo clean will delete".
    #[test]
    fn cargo_target_is_recognised_by_its_marker_not_its_name() {
        let tmp = std::env::temp_dir().join(format!("pd-dr-{}", std::process::id()));
        let target = tmp.join("target");
        let release = target.join("release");
        std::fs::create_dir_all(&release).unwrap();

        // A folder called target/release with no marker is just a folder: a
        // user is entitled to unzip paddock into one and get portable mode.
        assert!(!in_cargo_target(&release));

        std::fs::write(target.join("CACHEDIR.TAG"), b"Signature: 8a477f597d28d172").unwrap();
        assert!(in_cargo_target(&release));

        // And the marker only counts one level up - the exe's own directory
        // holding one would mean something else entirely.
        let stray = tmp.join("elsewhere");
        std::fs::create_dir_all(&stray).unwrap();
        std::fs::write(stray.join("CACHEDIR.TAG"), b"Signature: 8a477f597d28d172").unwrap();
        assert!(!in_cargo_target(&stray));

        std::fs::remove_dir_all(&tmp).ok();
    }
}
